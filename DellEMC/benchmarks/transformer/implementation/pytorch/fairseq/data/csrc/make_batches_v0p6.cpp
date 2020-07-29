#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace at {
namespace native {

namespace {

int64_t roundup(int64_t x, int64_t multiple)
{
    return (x + multiple - 1) / multiple * multiple;
} // roundup

int64_t rounddown(int64_t x, int64_t multiple)
{
    return x / multiple * multiple;
} // rounddown

std::pair<std::vector<int64_t>, std::vector<int64_t>> create_bucket_bounds_lists(
    int64_t max_allowable_seq_length,
    int64_t bucket_specify_min_boundary,
    float bucket_specify_growth_scale,
    bool use_efficient_last_pack)
{
    std::vector<int64_t> bucket_boundaries;

    auto x = bucket_specify_min_boundary;
    while (x < max_allowable_seq_length)
    {
        bucket_boundaries.emplace_back(x);
        x = std::max(x + 1, static_cast<int64_t>(x * bucket_specify_growth_scale));
    }

    std::vector<int64_t> buckets_min_list;
    buckets_min_list.emplace_back(0);
    std::vector<int64_t> buckets_max_list;
    if (use_efficient_last_pack)
    {
        for (auto && bound : bucket_boundaries)
        {
            buckets_min_list.emplace_back(bound + 1);
            buckets_max_list.emplace_back(bound);
        }

        buckets_max_list.push_back(max_allowable_seq_length);
    }
    else
    {
        for (auto && bound : bucket_boundaries)
        {
            buckets_min_list.emplace_back(bound);
            buckets_max_list.emplace_back(bound);
        }

        buckets_max_list.emplace_back(max_allowable_seq_length + 1);
    }

    return std::make_pair(buckets_min_list, buckets_max_list);
} // create_bucket_bounds_lists

int64_t seq_len_to_bucket_idx(
    int64_t seq_length,
    int64_t max_seq_length,
    std::vector<int64_t> buckets_min_list,
    std::vector<int64_t> buckets_max_list)
{
    int64_t idx = 0;

    // TODO: Update to bisection if execution time actually matters (avoiding premature optimization)
    // TODO: Alternate is to make lookup table keys = [0...256] and values are the buckets (this loop just indexes in to avoid repeated traversals)
    if (seq_length <= max_seq_length)
    {
        while (idx < static_cast<int64_t>(buckets_min_list.size()) && !(buckets_min_list[idx] <= seq_length && seq_length < buckets_max_list[idx]))
        {
            ++idx;
        }
    }
    else
    {
        idx = -1;
    }

    return idx;
} // seq_len_to_bucket_idx

int64_t seq_len_to_bucket_idx_improved_pack(
    int64_t seq_length,
    int64_t max_seq_length,
    std::vector<int64_t> buckets_min_list,
    std::vector<int64_t> buckets_max_list)
{
    int64_t idx = 0;

    // TODO: Update to bisection if execution time actually matters (avoiding premature optimization)
    // TODO: Alternate is to make lookup table keys = [0...256] and values are the buckets (this loop just indexes in to avoid repeated traversals)
    if (seq_length <= max_seq_length)
    {
        while (idx < static_cast<int64_t>(buckets_min_list.size()) && !(buckets_min_list[idx] <= seq_length && seq_length <= buckets_max_list[idx]))
        {
            ++idx;
        }
    }
    else
    {
        idx = -1;
    }

    return idx;
} // seq_len_to_bucket_idx_improved_pack

std::pair<std::vector<int64_t>, std::vector<int64_t>> create_seq_to_bucket_id_list_and_n_seq_per_batch(
    std::vector<int64_t> n_tok_per_seq,             // Number of tokens per sequence
    int64_t max_allowable_seq_length,               // Maximum sequence length to be considered (rejected if over)
    int64_t max_tokens,                             // Maximum number of tokens allowed in the batch
    int64_t pad_seq_per_batch_to_multiple_of,       // Padding multiple required, for number of sequences in batch
    int64_t pad_tok_per_seq_to_multiple_of,         // Padding multiple required, for number of tokens for sequence
    int64_t bucket_specify_min_boundary,            // This is the first non-zero beginning of a bucket (zero implicitly added)
    float bucket_specify_growth_scale,              // The next bucket bound is determined from the previous based on this factor
    bool do_seq_len_padding_to_multiple,            // Switch, enables padding sequence length to multiple
    bool do_batch_size_rounding_down_to_multiple,   // Switch, enables making other dimension of batch a multiple, based on number of sequences
    bool do_dynamic_batch_size_choice,              // Switch, enables choosing between methods on a batch-by-batch basis for efficiency
    bool use_efficient_last_pack)                   // Switch, modifies bucket bounds logic to improve batching
{
    const auto min_max_bounds = create_bucket_bounds_lists(
        max_allowable_seq_length,
        bucket_specify_min_boundary,
        bucket_specify_growth_scale,
        use_efficient_last_pack);

    std::vector<int64_t> n_seq_per_batch;
    std::vector<int64_t> bucket_idx_list;

    const auto bucket_interval_min = min_max_bounds.first;
    const auto bucket_interval_max = min_max_bounds.second;

    // Choose method
    if (do_seq_len_padding_to_multiple)
    {
        for (auto && item : bucket_interval_max)
        {
            n_seq_per_batch.emplace_back(max_tokens / roundup(item, pad_tok_per_seq_to_multiple_of));
        }
    }
    else if (do_batch_size_rounding_down_to_multiple)
    {
        for (auto && item : bucket_interval_max)
        {
            n_seq_per_batch.emplace_back(rounddown(max_tokens / item, pad_seq_per_batch_to_multiple_of));
        }
    }
    else if (do_dynamic_batch_size_choice)
    {
        for (auto && item : bucket_interval_max)
        {
            auto option1 = max_tokens / roundup(item, pad_tok_per_seq_to_multiple_of);
            auto option2 = rounddown(max_tokens / item, pad_seq_per_batch_to_multiple_of);
            n_seq_per_batch.emplace_back(std::max(option1, option2));
        }
    }
    else
    {
        for (auto && item : bucket_interval_max)
        {
            n_seq_per_batch.emplace_back(max_tokens / item);
        }
    }

    // Choose more efficient bounds
    if (use_efficient_last_pack)
    {
        for (auto && seq_length : n_tok_per_seq)
        {
            int64_t bucket_idx = seq_len_to_bucket_idx_improved_pack(seq_length, max_allowable_seq_length, bucket_interval_min, bucket_interval_max);

            bucket_idx_list.push_back(bucket_idx);
        }
    }
    else
    {
        for (auto && seq_length : n_tok_per_seq)
        {
            auto bucket_idx = seq_len_to_bucket_idx(seq_length, max_allowable_seq_length, bucket_interval_min, bucket_interval_max);

            bucket_idx_list.emplace_back(bucket_idx);
        }
    }

    return std::make_pair(bucket_idx_list, n_seq_per_batch);
} // create_seq_to_bucket_id_list_and_n_seq_per_batch


std::vector<std::vector<int64_t> > make_batches_v0p6(
    py::array_t<int64_t> src_lengths,
    py::array_t<int64_t> tgt_lengths,
    py::array_t<int64_t> idx_list,
    int64_t max_tokens,
    int64_t max_sentences,
    int64_t bsz_mult,
    int64_t max_len,
    int64_t bucket_specify_min_boundary,
    float bucket_specify_growth_scale,
    int64_t batch_strategy,
    bool use_efficient_last_pack)
{
    auto src_l = src_lengths.unchecked<1>();
    auto tgt_l = tgt_lengths.unchecked<1>();
    auto idx_l = idx_list.unchecked<1>();

    std::vector<std::vector<int64_t> > batches(1);

    std::vector<int64_t> n_tok_per_seq;

    for (int64_t i = 0; i < src_l.shape(0); ++i)
    {
        const int64_t src_len = src_l(i);
        const int64_t tgt_len = tgt_l(i);

        n_tok_per_seq.emplace_back(std::max(src_len, tgt_len));
    }

    const bool do_seq_len_padding_to_multiple = batch_strategy == 1;
    const bool do_batch_size_rounding_down_to_multiple = batch_strategy == 0;
    const bool do_dynamic_batch_size_choice = batch_strategy == 2;

    // Get vector of bucket ids (one per seq)
    const auto bucket_ids_and_n_seq_per_batch = create_seq_to_bucket_id_list_and_n_seq_per_batch(
        n_tok_per_seq,
        max_len,
        max_tokens,
        bsz_mult,
        bsz_mult,  // TODO: Make this independently varied (for now assumed to be 8 for both anyways)
        bucket_specify_min_boundary,
        bucket_specify_growth_scale,
        do_seq_len_padding_to_multiple,
        do_batch_size_rounding_down_to_multiple,
        do_dynamic_batch_size_choice,
        use_efficient_last_pack);

     const auto bucket_ids = bucket_ids_and_n_seq_per_batch.first;
     const auto n_seq_per_batch = bucket_ids_and_n_seq_per_batch.second;

    // Get buckets
    const auto min_max_bounds = create_bucket_bounds_lists(
        max_len,
        bucket_specify_min_boundary,
        bucket_specify_growth_scale,
        use_efficient_last_pack);

    const auto bucket_interval_min = min_max_bounds.first;
    const auto bucket_interval_max = min_max_bounds.second;

    // Fill buckets
    std::vector<std::vector<int64_t> > buckets(bucket_interval_min.size(), std::vector<int64_t>());

    int64_t id_cnt = 0;
    for (auto && id : bucket_ids)
    {
        if (id == -1)
        {
            id_cnt += 1;
        }
    }

    int64_t reject_count = 0;
    int64_t dummy = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(bucket_ids.size()); ++i)
    {
        if (bucket_ids[i] >= 0)
        {
            const auto bidx = bucket_ids[i];
            buckets[bidx].emplace_back(i);
        }
        else
        {
            ++reject_count;
        }
    }

    // Get number sequences rejected due to sequence length
    std::cout << reject_count << " sequences were omitted due to containing over " << max_len << " tokens." << std::endl;

    int64_t batch_n_seq = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(buckets.size()); ++i)
    {
        const auto bucket = buckets[i];
        const auto nspb = n_seq_per_batch[i];
	const auto  bkt_max_len = bucket_interval_max[i];

        for (auto && item : bucket)
        {
            if (batch_n_seq < nspb)
            {
                batches.back().emplace_back(item);
                ++batch_n_seq;

            }
            else
            {
                std::vector<int64_t> new_batch;
                new_batch.emplace_back(item);
                batches.emplace_back(new_batch);
                batch_n_seq = 1;
            }
        }
	auto &last_batch = batches.back();
	if (last_batch.size() % bsz_mult != 0) {
	    auto batch_size = last_batch.size();
	    auto max_len    = std::max(last_batch.begin(), last_batch.end());
	    auto tokens     = batch_size * roundup(bkt_max_len, bsz_mult);
	    if (tokens > max_tokens) {
		auto half_batch = batch_size / 2;
		std::vector<int64_t> new_batch;
		new_batch.reserve(half_batch);
		std::copy(last_batch.begin() + half_batch, last_batch.end(), std::back_inserter(new_batch));
		last_batch.erase(last_batch.begin() + half_batch, last_batch.end());
		batches.emplace_back(new_batch);
	    }
	}

        batches.emplace_back(std::vector<int64_t>());
        batch_n_seq = 0;
    }

    if (batches.back().empty())
    {
        batches.pop_back();
    }

    auto i = std::begin(batches);

    while (i != std::end(batches)) 
    {
        if ((*i).empty())
        {
            i = batches.erase(i);
        }
        else
        {
            ++i;
        }
    }

    return batches;
    }
} // namespace

} // namespace at
} // namespace native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("make_batches_v0p6", &at::native::make_batches_v0p6);
}

