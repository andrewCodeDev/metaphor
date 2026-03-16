// Test: verify that @subgraph's panic guard correctly restores current_graph
// when a panic is caught by an external recovery mechanism (e.g. the test
// runner's setjmp/longjmp).
//
// Each @subgraph sets up its own setjmp frame. On panic, it cleans up its
// graph, restores current_graph to the parent, and re-raises. The tests
// verify this by catching the re-raised panic and checking graph state.
module metaphor::testing::graph::test_subgraph_panic_guard;
import metaphor::graph;
import libc;

alias PanicFn = fn void(String message, String file, String function, uint line);

// Our test-level panic handler — longjmps back to the test
tlocal libc::JmpBuf test_jmpbuf;

fn void test_panic_catcher(String message, String file, String function, uint line)
{
	libc::longjmp(&test_jmpbuf, 1);
}

// ============================================================================
// Test 1: Normal subgraph — no panic, graph restored correctly
// ============================================================================

fn void test_subgraph_normal_exit() @test
{
	graph::Graph* before = graph::get();

	graph::@subgraph()
	{
		graph::Graph* inside = graph::get();
		assert(inside != before, "subgraph should be a different graph");
	};

	graph::Graph* after = graph::get();
	assert(after == before, "graph should be restored after subgraph");
}

// ============================================================================
// Test 2: Nested subgraphs — all restore correctly
// ============================================================================

fn void test_subgraph_nested_normal() @test
{
	graph::Graph* g0 = graph::get();

	graph::@subgraph()
	{
		graph::Graph* g1 = graph::get();
		assert(g1 != g0);

		graph::@subgraph()
		{
			graph::Graph* g2 = graph::get();
			assert(g2 != g1);
			assert(g2 != g0);
		};

		assert(graph::get() == g1, "inner subgraph should restore to g1");
	};

	assert(graph::get() == g0, "outer subgraph should restore to g0");
}

// ============================================================================
// Test 3: Panic inside subgraph — graph restored
// ============================================================================

fn void test_subgraph_panic_restores_graph() @test
{
	graph::Graph* before = graph::get();

	// Install our own panic catcher to recover from the re-raised panic
	PanicFn saved_panic = builtin::panic;
	builtin::panic = &test_panic_catcher;

	if (libc::setjmp(&test_jmpbuf) == 0)
	{
		graph::@subgraph()
		{
			graph::Graph* inside = graph::get();
			assert(inside != before, "should be in subgraph");

			// Trigger panic — subgraph catches, cleans up, re-raises
			builtin::panic("deliberate panic in subgraph", $$FILE, $$FUNC, $$LINE);
		};
	}

	// Restore our panic handler
	builtin::panic = saved_panic;

	// The subgraph's cleanup should have restored current_graph
	graph::Graph* after = graph::get();
	assert(after == before, "graph must be restored after panic");
}

// ============================================================================
// Test 4: Panic in nested subgraph — each level unwinds, graph restored to root
// ============================================================================

fn void test_subgraph_nested_panic_restores_root() @test
{
	graph::Graph* g0 = graph::get();

	PanicFn saved_panic = builtin::panic;
	builtin::panic = &test_panic_catcher;

	if (libc::setjmp(&test_jmpbuf) == 0)
	{
		graph::@subgraph()
		{
			graph::@subgraph()
			{
				graph::@subgraph()
				{
					builtin::panic("panic at depth 3", $$FILE, $$FUNC, $$LINE);
				};
			};
		};
	}

	builtin::panic = saved_panic;

	// Each level cleaned up and re-raised — should be back to root
	assert(graph::get() == g0, "graph must be restored to root after nested panic");
}

// ============================================================================
// Test 5: Nested panic recovery — graph and panic handler fully restored
// ============================================================================

fn void test_subgraph_nested_panic_full_restore() @test
{
	graph::Graph* g0 = graph::get();

	PanicFn saved_panic = builtin::panic;
	builtin::panic = &test_panic_catcher;

	// Panic at depth 3
	if (libc::setjmp(&test_jmpbuf) == 0)
	{
		graph::@subgraph()
		{
			graph::@subgraph()
			{
				graph::@subgraph()
				{
					builtin::panic("panic at depth 3", $$FILE, $$FUNC, $$LINE);
				};
			};
		};
	}

	builtin::panic = saved_panic;

	// Graph pointer restored
	assert(graph::get() == g0, "graph must be restored to root");

	// Panic handler restored
	assert(builtin::panic == saved_panic, "panic handler must be restored");
}
