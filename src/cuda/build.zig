const std = @import("std");

/// Required stub for Zig build system (this file is not used as a standalone build)
pub fn build(_: *std.Build) void {}

// NOTE: This file is deprecated. The main build.zig has its own make_cuda_module.
// Keeping this file to avoid breaking anything that might reference it.
