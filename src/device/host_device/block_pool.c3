//! Block Pool — buddy allocator for device memory
//!
//! Arena-backed buddy allocator that supports block splitting and fusion.
//! Pools always allocate a fixed region via mmap (vm::alloc). Blocks are
//! managed in a 2D web: split-siblings (left/right contiguous memory) and
//! order-siblings (same-order free list). This gives O(1) alloc from a
//! matching order and O(1) merge on free.
//!

module metaphor::device::host_device::block_pool;
import std::core::mem::allocator;
import std::core::mem::vm;
import metaphor::utils::hash_map;

// Constants

const usz MIN_ORDER  = 3;   // 2^3  = 8 bytes (supports scalar f64)
const usz MAX_ORDER  = 32;  // 2^32 = 4 GB
const usz NUM_ORDERS = MAX_ORDER - MIN_ORDER + 1; // 30

// PtrKey — HashMap key wrapper for pointer addresses

struct PtrKey
{
	usz value;
}

fn uint PtrKey.hash(&self)
{
	// xor-shift hash for pointer addresses
	usz h = self.value;
	h = (h ^ (h >> 16)) * 0x45d9f3b;
	h = (h ^ (h >> 16)) * 0x45d9f3b;
	h = h ^ (h >> 16);
	return (uint)h;
}

fn bool PtrKey.equals(self, PtrKey other) @operator(==)
{
	return self.value == other.value;
}

// Block — represents a contiguous memory region

struct Block
{
	char[] data;          // the memory region this block covers
	bool used;
	Block* split_prev;    // left buddy sibling (contiguous memory)
	Block* split_next;    // right buddy sibling (contiguous memory)
	Block* order_prev;    // doubly-linked order list
	Block* order_next;    // doubly-linked order list
}

// OrderList — doubly-linked free list per order

struct OrderList
{
	Block* first;
}

fn void OrderList.prepend(&self, Block* block)
{
	if (self.first)
	{
		self.first.order_prev = block;
	}
	block.order_next = self.first;
	block.order_prev = null;
	self.first = block;
}

fn void OrderList.remove(&self, Block* block)
{
	if (block.order_prev)
	{
		block.order_prev.order_next = block.order_next;
	}
	else
	{
		// First element
		self.first = block.order_next;
	}

	if (block.order_next)
	{
		block.order_next.order_prev = block.order_prev;
	}
}

fn Block* OrderList.pop(&self)
{
	Block* first = self.first;
	if (!first) return null;
	self.remove(first);
	return first;
}

// BlockMap — maps outgoing slice pointers to Block* for O(1) free lookup

alias BlockMap = hash_map::HashMap{PtrKey, Block*};

// Faults

faultdef OVERFLOW, FRAGMENTED, EXCEEDS_MAX;

// BlockPool

struct BlockPool
{
	char[] mem_buf;                       // mmap'd region
	usz mem_rem;                          // remaining bytes
	usz peak_val;                         // high-water mark
	usz index_sentinel;                   // highest non-empty order index + 1
	OrderList[NUM_ORDERS] block_orders;   // free blocks by order
	DynamicArenaAllocator block_arena;    // metadata allocation
	Block* free_head;                     // freed block metadata chain
	BlockMap block_map;                   // ptr -> Block* for O(1) free
	Block* top_block;                     // root block (invariant for reset)
}

// Lifecycle

fn BlockPool*? BlockPool.init(&self, usz pool_size)
{
	void* raw = vm::alloc(pool_size, READWRITE)!;
	self.mem_buf = ((char*)raw)[:pool_size];
	self.mem_rem = pool_size;
	self.index_sentinel = min(order_to_index(size_to_lower_order(pool_size)) + 1, NUM_ORDERS);
	for (usz i = 0; i < NUM_ORDERS; i++)
	{
		self.block_orders[i] = {};
	}
	self.block_arena.init(mem, 4096);
	self.free_head = null;
	self.block_map.init(mem);

	self.top_block = self.create_block()!;
	self.top_block.data = self.mem_buf;
	self.top_block.used = true;
	self.top_block.split_prev = null;
	self.top_block.split_next = null;
	self.top_block.order_prev = null;
	self.top_block.order_next = null;

	self.reserve_block(self.top_block);

	return self;
}

fn void BlockPool.deinit(&self)
{
	self.block_map.free();
	self.block_arena.free();
	vm::release(self.mem_buf.ptr, self.mem_buf.len)!!;
}

// Alloc / Free / Reset

fn char[]? BlockPool.alloc(&self, usz byte_size)
{
	usz min_size = order_to_size(MIN_ORDER);
	usz max_size = order_to_size(MAX_ORDER);

	// Round up to multiple of min_size
	usz split_size = ((byte_size + min_size - 1) / min_size) * min_size;
	if (split_size < min_size) split_size = min_size;

	usz upper_order = size_to_upper_order(split_size);

	if (self.mem_rem < byte_size)
	{
		return OVERFLOW~;
	}
	if (max_size < byte_size)
	{
		return EXCEEDS_MAX~;
	}

	// Scan up to and including the max reserved order
	Block* lhs = null;
	for (usz i = order_to_index(upper_order); i < self.index_sentinel; i++)
	{
		Block* rhs = self.release_block(i);
		if (!rhs) continue;

		// No need to split
		if (split_size == rhs.data.len)
		{
			lhs = rhs;
			break;
		}

		Block* new_block = self.create_block()!;
		usz rhs_data_len = rhs.data.len;

		// Set up the new left block
		new_block.data = rhs.data[:split_size];
		new_block.used = true;
		new_block.split_prev = rhs.split_prev;
		new_block.split_next = rhs;
		new_block.order_prev = null;
		new_block.order_next = null;

		// Fix up the previous block's next pointer
		if (rhs.split_prev)
		{
			rhs.split_prev.split_next = new_block;
		}

		// Connect rhs back to new_block
		rhs.split_prev = new_block;

		// Shrink rhs to the remainder
		rhs.data = rhs.data[split_size..];
		self.reserve_block(rhs);

		lhs = new_block;
		break;
	}

	if (!lhs)
	{
		return FRAGMENTED~;
	}

	self.mem_rem -= lhs.data.len;
	usz current = self.mem_buf.len - self.mem_rem;
	if (current > self.peak_val) { self.peak_val = current; }

	char[] result = lhs.data[:byte_size];

	// Register block in map for O(1) free lookup
	self.block_map.set({ .value = (usz)lhs.data.ptr }, lhs);

	return result;
}

fn void BlockPool.free(&self, char[] slice)
{
	PtrKey key = { .value = (usz)slice.ptr };
	Block* block = self.block_map.get(key) ?? null;
	assert(block != null, "free: block not found for pointer");

	self.block_map.remove(key)!!;
	self.mem_rem += block.data.len;

	// Fuse left side
	Block* fused = block;
	if (block.split_prev)
	{
		Block* left = block.split_prev;
		if (!left.used)
		{
			self.disconnect_from_order(left);

			if (block.split_next)
			{
				block.split_next.split_prev = left;
			}
			left.split_next = block.split_next;
			left.data = left.data.ptr[:left.data.len + block.data.len];
			self.free_block(block);
			fused = left;
		}
	}

	// Fuse right side
	if (fused.split_next)
	{
		Block* right = fused.split_next;
		if (!right.used)
		{
			self.disconnect_from_order(right);

			if (right.split_next)
			{
				right.split_next.split_prev = fused;
			}
			fused.split_next = right.split_next;
			fused.data = fused.data.ptr[:fused.data.len + right.data.len];
			self.free_block(right);
		}
	}

	self.reserve_block(fused);
}

fn void BlockPool.reset(&self)
{
	for (usz i = 0; i < NUM_ORDERS; i++)
	{
		self.block_orders[i] = {};
	}

	self.free_head = null;
	self.block_map.clear();
	self.block_arena.reset();

	// Re-allocate top_block since arena was reset
	self.top_block = allocator::new_try(&self.block_arena, Block)!!;
	self.top_block.data = self.mem_buf;
	self.top_block.used = true;
	self.top_block.split_prev = null;
	self.top_block.split_next = null;
	self.top_block.order_prev = null;
	self.top_block.order_next = null;

	self.mem_rem = self.mem_buf.len;
	self.peak_val = 0;
	self.index_sentinel = min(order_to_index(size_to_lower_order(self.mem_buf.len)) + 1, NUM_ORDERS);

	self.reserve_block(self.top_block);
}

// Memory tracking

fn usz BlockPool.mem_used(&self) => self.mem_buf.len - self.mem_rem;

fn usz BlockPool.mem_reserved(&self) => self.mem_buf.len;

fn usz BlockPool.mem_peak(&self) => self.peak_val;

// Internal helpers

fn void BlockPool.disconnect_from_order(&self, Block* block)
{
	usz index = size_to_lower_index(block.data.len);
	self.block_orders[index].remove(block);
}

fn Block*? BlockPool.create_block(&self)
{
	if (self.free_head)
	{
		Block* block = self.free_head;
		self.free_head = block.order_next;
		return block;
	}
	return allocator::new_try(&self.block_arena, Block);
}

fn void BlockPool.free_block(&self, Block* block)
{
	// Chain freed blocks via order_next (safe: freed metadata isn't in any order list)
	block.order_next = self.free_head;
	self.free_head = block;
}

fn void BlockPool.reserve_block(&self, Block* block)
{
	usz index = size_to_lower_index(block.data.len);
	self.block_orders[index].prepend(block);
	block.used = false;
}

fn Block* BlockPool.release_block(&self, usz index)
{
	Block* block = self.block_orders[index].pop();
	if (!block) return null;
	block.used = true;
	return block;
}

// Order/size conversion helpers (module-level)

fn usz order_to_size(usz order)
{
	assert(order >= MIN_ORDER && order <= MAX_ORDER);
	return (usz)1 << (usz)order;
}

fn usz size_to_upper_order(usz size)
{
	usz order = MIN_ORDER;
	usz bsize = (usz)1 << MIN_ORDER;
	while (bsize < size && order < MAX_ORDER)
	{
		bsize <<= 1;
		order += 1;
	}
	return order;
}

fn usz size_to_lower_order(usz size)
{
	usz order = MIN_ORDER;
	for (usz o = MIN_ORDER; o <= MAX_ORDER; o++)
	{
		usz order_size = (usz)1 << o;
		if (size < order_size)
		{
			return max(o - 1, MIN_ORDER);
		}
	}
	return MAX_ORDER;
}

fn usz order_to_index(usz order)
{
	assert(order >= MIN_ORDER);
	return order - MIN_ORDER;
}

fn usz index_to_order(usz index)
{
	assert(index < NUM_ORDERS);
	return index + MIN_ORDER;
}

fn usz size_to_lower_index(usz size)
{
	return order_to_index(size_to_lower_order(size));
}

fn usz min(usz a, usz b) @inline => a < b ? a : b;
fn usz max(usz a, usz b) @inline => a > b ? a : b;
