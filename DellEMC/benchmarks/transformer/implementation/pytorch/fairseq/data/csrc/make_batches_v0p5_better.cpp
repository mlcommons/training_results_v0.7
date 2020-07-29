#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

namespace at {
namespace native {

// In lieu of a header file...
enum BatchingScheme
{
  MAKE_BATCHES_V0P5_BETTER = 0,
  MAKE_BATCHES_V0P5_EVEN_BETTER = 1
};

int64_t roundup(int64_t x, int64_t multiple);
int64_t rounddown(int64_t x, int64_t multiple);

template<int BatchingScheme> std::vector<std::vector<int64_t> > make_batches(
    py::array_t<int64_t> src_lengths,
	py::array_t<int64_t> tgt_lengths,
	py::array_t<int64_t> idx_list,
	int64_t max_tokens,
	int64_t max_sentences,
	int64_t max_len,
	int64_t bsz_mult,
	int64_t pad_seq);

// Source starts here
int64_t roundup(int64_t x, int64_t multiple) {
  return (x + multiple - 1) / multiple * multiple;
} // roundup


int64_t rounddown(int64_t x, int64_t multiple)
{
    return x / multiple * multiple;
} // rounddown


bool is_batch_full(int64_t num_tokens, int64_t max_tokens, int64_t max_sentences, int64_t batch_length){
  if (batch_length == 0)
  {
    return false;
  }
  else if (batch_length == max_sentences || num_tokens > max_tokens)
  {
    return true;
  }
  else
  {
    return false;
  }
} // is_batch_full


template <>
std::vector<std::vector<int64_t>> make_batches<MAKE_BATCHES_V0P5_BETTER>(
    py::array_t<int64_t> src_lengths,
	py::array_t<int64_t> tgt_lengths,
	py::array_t<int64_t> idx_list,
	int64_t max_tokens,
	int64_t max_sentences,
	int64_t max_len,
	int64_t bsz_mult,
	int64_t pad_seq)
{
   std::vector<std::vector<int64_t>> batches;

   const auto src_l = src_lengths.unchecked<1>();
   const auto tgt_l = tgt_lengths.unchecked<1>();
   const auto idx_l = idx_list.unchecked<1>();

   AT_ASSERTM(src_l.shape(0) == tgt_l.shape(0), "tgt_list and src_list should have the same shape");
   AT_ASSERTM(idx_l.shape(0) == tgt_l.shape(0), "idx_list and tgt_list should have the same shape");

   const auto nelem = src_l.shape(0);
   int64_t sample_len = 0;
   int64_t padded_sample_len = 0;

   const auto num_seqs_mult = ((bsz_mult % pad_seq) == 0) ? bsz_mult / pad_seq : bsz_mult;

   std::vector<int64_t> sample_lens;
   std::vector<int64_t> batch;

   for (ssize_t i = 0; i < nelem; ++i){
       const auto idx = idx_l(i);
       const auto sample_num_tokens = std::max(src_l(idx), tgt_l(idx));

       if (sample_num_tokens > max_len) continue; 

       sample_len = std::max(sample_len, sample_num_tokens);
       padded_sample_len = (static_cast<int64_t>(batch.size()) < num_seqs_mult) ? roundup(sample_len, bsz_mult) : roundup(sample_len, pad_seq);
       sample_lens.emplace_back(sample_num_tokens);
       int64_t num_tokens = (batch.size() + 1) * padded_sample_len;

       if (is_batch_full(num_tokens, max_tokens, max_sentences, batch.size()))
       {
          auto sequences = batch.size();
          if ( ((sequences % num_seqs_mult) != 0) && (sequences > num_seqs_mult) ) {
            auto pad_sequences_opt_seqs  = rounddown(sequences, num_seqs_mult);
            auto total_tokens_opt_seqs   = padded_sample_len * pad_sequences_opt_seqs;
         
            auto pad_seq_len_opt_seqlen  = roundup(padded_sample_len, bsz_mult);
            auto pad_sequences_opt_seqlen= max_tokens / pad_seq_len_opt_seqlen;
            auto total_tokens_opt_seqlen = padded_sample_len * pad_sequences_opt_seqlen;
            
            if(total_tokens_opt_seqs >= total_tokens_opt_seqlen) {
              sequences = pad_sequences_opt_seqs;
            } else {
              sequences = pad_sequences_opt_seqlen;
            } 
          }
          //std::cout << "BATCH: Sentences: " << sequences << " Sent Length: " << sample_len << " Total: " << sample_len*sequences << " " << (static_cast<float>(sample_len * sequences) / static_cast<float>(max_tokens) * 100.0) << std::endl;

          std::vector<int64_t> new_batch;
          new_batch.reserve(sequences);
          std::copy(batch.begin() + sequences, batch.end(), std::back_inserter(new_batch));
          batch.erase(batch.begin() + sequences, batch.end());
          sample_lens.erase(sample_lens.begin(), sample_lens.begin() + sequences);
          sample_len = *std::max_element(sample_lens.begin(), sample_lens.end());
          batches.emplace_back(batch);
          batch = new_batch;
       }

       batch.emplace_back(idx);
   }

   while (batch.size() > 0)
   {
     const auto sequences = std::max(batch.size() / num_seqs_mult * num_seqs_mult, batch.size() % num_seqs_mult);
     std::vector<int64_t> new_batch;
     new_batch.reserve(sequences);
     std::copy(batch.begin() + sequences, batch.end(), std::back_inserter(new_batch));
     batch.erase(batch.begin() + sequences, batch.end());
     batches.emplace_back(batch);
     batch = new_batch;
   }

   return batches;
} // make_batches<MAKE_BATCHES_V0P5_BETTER>


template <>
std::vector<std::vector<int64_t>> make_batches<MAKE_BATCHES_V0P5_EVEN_BETTER>(
    py::array_t<int64_t> src_lengths,
	py::array_t<int64_t> tgt_lengths,
	py::array_t<int64_t> idx_list,
	int64_t max_tokens,
	int64_t max_sentences,
	int64_t max_len,
	int64_t bsz_mult,
	int64_t pad_seq)
{
   std::vector<std::vector<int64_t> > batches(1);

   const auto src_l = src_lengths.unchecked<1>();
   const auto tgt_l = tgt_lengths.unchecked<1>();
   const auto idx_l = idx_list.unchecked<1>();

   AT_ASSERTM(src_l.shape(0) == tgt_l.shape(0), "tgt_list and src_list should have the same shape");
   AT_ASSERTM(idx_l.shape(0) == tgt_l.shape(0), "idx_list and tgt_list should have the same shape");

   // argsort
   std::vector<int64_t> max_lengths(src_l.size());
   for (int64_t i = 0; i < src_l.size(); ++i)
   {
     max_lengths[i] = std::max(src_l[i], tgt_l[i]);
   }

   std::vector<int64_t> perm(src_l.size());
   iota(perm.begin(), perm.end(), 0);
   std::sort(
     perm.begin(),
     perm.end(),
     [&max_lengths](int64_t i1, int64_t i2) { return max_lengths[i1] > max_lengths[i2]; }); // descending order

   int64_t offset = 0;
   while (max_lengths[perm[offset]] > max_len) // skip all sequences over specified length
   {
     ++offset;
   }
   int64_t padded_seq_len = roundup(max_lengths[perm[offset]], pad_seq);
   int64_t max_seq_in_batch = max_tokens / padded_seq_len;
   int64_t n_seq_in_batch = 0;
   int64_t n_tok_in_batch = 0;

   for (auto && i : perm)
   {
     if (max_lengths[i] > max_len) continue;

     if (n_tok_in_batch + padded_seq_len < max_tokens && n_seq_in_batch < rounddown(max_sentences, bsz_mult))
     {
       batches.back().emplace_back(i);
       ++n_seq_in_batch;
       n_tok_in_batch += padded_seq_len;
     }
     else
     {
       batches.emplace_back(std::vector<int64_t>(1, i));
       padded_seq_len = roundup(max_lengths[i], pad_seq);
       max_seq_in_batch = max_tokens / padded_seq_len;
       n_seq_in_batch = 1;
       n_tok_in_batch = padded_seq_len;
     }
   }

   return batches;
} // make_batches<MAKE_BATCHES_V0P5_EVEN_BETTER>


} // namespace native
} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("make_batches_v0p5_better", &at::native::make_batches<at::native::MAKE_BATCHES_V0P5_BETTER>); // Relying on this line for instantiation
} // PYBIND11_MODULE
