#ifndef CC_SAMPLE_RECORDS_H
#define CC_SAMPLE_RECORDS_H

#include <string>
#include <list>

namespace minigo {
namespace sample_records {

int32_t run(double sample_fraction,
            uint64_t num_records,
            int32_t num_read_threads,
            int32_t num_write_threads,
            int32_t compression,
            int32_t files_per_pattern,
            bool shuffle,
            uint64_t seed,
            std::list<std::string> src_patterns,
            std::string dst,
            bool verbose);

} // namespace sample_records 
} // namespace minigo

#endif // CC_SAMPLE_RECORDS_H
