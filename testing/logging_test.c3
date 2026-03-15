module metaphor::testing::logging_test;
import metaphor::logging;

fn void test_default_all_disabled() @test
{
	// All scopes disabled by default
	assert(!logging::is_enabled(Scope.GRAPH, Level.DEBUG));
	assert(!logging::is_enabled(Scope.TENSOR, Level.INFO));
	assert(!logging::is_enabled(Scope.DEVICE, Level.ERR));
}

fn void test_enable_single_scope() @test
{
	logging::disable_all();

	logging::enable(Scope.GRAPH, Level.INFO);
	assert(logging::is_enabled(Scope.GRAPH, Level.INFO));
	assert(logging::is_enabled(Scope.GRAPH, Level.WARN));
	assert(logging::is_enabled(Scope.GRAPH, Level.ERR));
	assert(!logging::is_enabled(Scope.GRAPH, Level.DEBUG), "DEBUG below min level INFO");

	// Other scopes still disabled
	assert(!logging::is_enabled(Scope.TENSOR, Level.INFO));

	logging::disable_all();
}

fn void test_enable_debug_allows_all() @test
{
	logging::disable_all();

	logging::enable(Scope.TENSOR, Level.DEBUG);
	assert(logging::is_enabled(Scope.TENSOR, Level.DEBUG));
	assert(logging::is_enabled(Scope.TENSOR, Level.INFO));
	assert(logging::is_enabled(Scope.TENSOR, Level.WARN));
	assert(logging::is_enabled(Scope.TENSOR, Level.ERR));

	logging::disable_all();
}

fn void test_disable_scope() @test
{
	logging::disable_all();

	logging::enable(Scope.DEVICE);
	assert(logging::is_enabled(Scope.DEVICE, Level.DEBUG));

	logging::disable(Scope.DEVICE);
	assert(!logging::is_enabled(Scope.DEVICE, Level.DEBUG));
}

fn void test_enable_all() @test
{
	logging::disable_all();

	logging::enable_all(Level.WARN);
	assert(logging::is_enabled(Scope.GRAPH, Level.WARN));
	assert(logging::is_enabled(Scope.TENSOR, Level.ERR));
	assert(!logging::is_enabled(Scope.GRAPH, Level.DEBUG));
	assert(!logging::is_enabled(Scope.TENSOR, Level.INFO));

	logging::disable_all();
}

fn void test_log_calls_dont_crash() @test
{
	// Should be no-ops when disabled
	logging::disable_all();
	logging::@debug(GRAPH, "should not appear");
	logging::@info(TENSOR, "should not appear");
	logging::@warn(DEVICE, "should not appear");
	logging::@err(ALLOCATOR, "should not appear");
}

