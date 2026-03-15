/*
 * logging.h - Simple logging for HIP code
 *
 * Usage:
 *   LOG_DEBUG("message: %d", value);
 *   LOG_INFO("tensor shape: [%lu, %lu]", x, y);
 *   LOG_WARN("unexpected value");
 *   LOG_ERROR("failed: %s", reason);
 *
 * Control:
 *   Define HIP_LOG_LEVEL before including to set minimum level:
 *     0 = DEBUG (all messages)
 *     1 = INFO
 *     2 = WARN
 *     3 = ERROR
 *     4 = NONE (silent)
 *
 *   Default is INFO (1) in release, DEBUG (0) when DEBUG is defined.
 */

#ifndef METAPHOR_HIP_LOGGING_H
#define METAPHOR_HIP_LOGGING_H

#include <cstdio>

/* Log levels */
#define HIP_LOG_LEVEL_DEBUG 0
#define HIP_LOG_LEVEL_INFO 1
#define HIP_LOG_LEVEL_WARN 2
#define HIP_LOG_LEVEL_ERROR 3
#define HIP_LOG_LEVEL_NONE 4

/* Default log level: WARN in release (compile with -DHIP_LOG_LEVEL=1 for INFO)
 */
#ifndef HIP_LOG_LEVEL
#define HIP_LOG_LEVEL HIP_LOG_LEVEL_WARN
#endif

/* Core logging macro */
#define HIP_LOG(level, level_str, fmt, ...)                                   \
  do {                                                                         \
    if (level >= HIP_LOG_LEVEL) {                                             \
      fprintf(stderr, "[HIP %s] %s:%d: " fmt "\n", level_str, __FILE__,       \
              __LINE__, ##__VA_ARGS__);                                        \
    }                                                                          \
  } while (0)

/* Level-specific macros */
#if HIP_LOG_LEVEL <= HIP_LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...)                                                    \
  HIP_LOG(HIP_LOG_LEVEL_DEBUG, "DEBUG", fmt, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) ((void)0)
#endif

#if HIP_LOG_LEVEL <= HIP_LOG_LEVEL_INFO
#define LOG_INFO(fmt, ...)                                                     \
  HIP_LOG(HIP_LOG_LEVEL_INFO, "INFO", fmt, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) ((void)0)
#endif

#if HIP_LOG_LEVEL <= HIP_LOG_LEVEL_WARN
#define LOG_WARN(fmt, ...)                                                     \
  HIP_LOG(HIP_LOG_LEVEL_WARN, "WARN", fmt, ##__VA_ARGS__)
#else
#define LOG_WARN(fmt, ...) ((void)0)
#endif

#if HIP_LOG_LEVEL <= HIP_LOG_LEVEL_ERROR
#define LOG_ERROR(fmt, ...)                                                    \
  HIP_LOG(HIP_LOG_LEVEL_ERROR, "ERROR", fmt, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) ((void)0)
#endif

/* Tensor logging helper (expects DenseCore*) */
#define LOG_TENSOR(name, t, syms)                                              \
  do {                                                                         \
    if (HIP_LOG_LEVEL <= HIP_LOG_LEVEL_DEBUG) {                              \
      fprintf(stderr,                                                          \
              "[HIP DEBUG] %s:%d: %s: dtype=%lu ndim=%lu numel=%lu data=%p "  \
              "shape=[",                                                       \
              __FILE__, __LINE__, name, (t)->dtype, (t)->shape.len,            \
              (t)->num_elements, (t)->data);                                   \
      for (len_t i = 0; i < (t)->shape.len; i++)                               \
        fprintf(stderr, "%lu%s", (t)->shape.buffer[i],                          \
                i + 1 < (t)->shape.len ? "," : "");                            \
      fprintf(stderr, "] syms=[");                                             \
      for (len_t i = 0; i < (t)->shape.len; i++)                               \
        fprintf(stderr, "%c%s", (char)(syms)[i],                               \
                i + 1 < (t)->shape.len ? "," : "");                            \
      fprintf(stderr, "]\n");                                                  \
    }                                                                          \
  } while (0)

#endif /* METAPHOR_HIP_LOGGING_H */
