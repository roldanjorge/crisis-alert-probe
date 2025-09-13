Model loaded successfully on cuda
Model config: HookedTransformerConfig:
{'NTK_by_parts_factor': 8.0,
 'NTK_by_parts_high_freq_factor': 4.0,
 'NTK_by_parts_low_freq_factor': 1.0,
 'act_fn': 'silu',
 'attention_dir': 'causal',
 'attn_only': False,
 'attn_scale': 11.313708498984761,
 'attn_scores_soft_cap': -1.0,
 'attn_types': None,
 'checkpoint_index': None,
 'checkpoint_label_type': None,
 'checkpoint_value': None,
 'd_head': 128,
 'd_mlp': 13824,
 'd_model': 5120,
 'd_vocab': 32000,
 'd_vocab_out': 32000,
 'decoder_start_token_id': None,
 'default_prepend_bos': True,
 'device': device(type='cuda'),
 'dtype': torch.float32,
 'eps': 1e-05,
 'experts_per_token': None,
 'final_rms': True,
 'from_checkpoint': False,
 'gated_mlp': True,
 'init_mode': 'gpt2',
 'init_weights': False,
 'initializer_range': 0.011180339887498949,
 'load_in_4bit': False,
 'model_name': 'Llama-2-13b-chat-hf',
 'n_ctx': 4096,
 'n_devices': 1,
 'n_heads': 40,
 'n_key_value_heads': None,
 'n_layers': 40,
 'n_params': 12687769600,
 'normalization_type': 'RMSPre',
 'num_experts': None,
 'original_architecture': 'LlamaForCausalLM',
 'output_logits_soft_cap': -1.0,
 'parallel_attn_mlp': False,
 'positional_embedding_type': 'rotary',
 'post_embedding_ln': False,
 'relative_attention_max_distance': None,
 'relative_attention_num_buckets': None,
 'rotary_adjacent_pairs': False,
 'rotary_base': 10000,
 'rotary_dim': 128,
 'scale_attn_by_inverse_layer_idx': False,
 'seed': None,
 'tie_word_embeddings': False,
 'tokenizer_name': 'meta-llama/Llama-2-13b-chat-hf',
 'tokenizer_prepends_bos': True,
 'trust_remote_code': True,
 'ungroup_grouped_query_attention': False,
 'use_NTK_by_parts_rope': False,
 'use_attn_in': False,
 'use_attn_result': False,
 'use_attn_scale': True,
 'use_hook_mlp_in': False,
 'use_hook_tokens': False,
 'use_local_attn': False,
 'use_normalization_before_and_after': False,
 'use_split_qkv_input': False,
 'window_size': None}
Model loaded successfully: HookedTransformerConfig:
{'NTK_by_parts_factor': 8.0,
 'NTK_by_parts_high_freq_factor': 4.0,
 'NTK_by_parts_low_freq_factor': 1.0,
 'act_fn': 'silu',
 'attention_dir': 'causal',
 'attn_only': False,
 'attn_scale': 11.313708498984761,
 'attn_scores_soft_cap': -1.0,
 'attn_types': None,
 'checkpoint_index': None,
 'checkpoint_label_type': None,
 'checkpoint_value': None,
 'd_head': 128,
 'd_mlp': 13824,
 'd_model': 5120,
 'd_vocab': 32000,
 'd_vocab_out': 32000,
 'decoder_start_token_id': None,
 'default_prepend_bos': True,
 'device': device(type='cuda'),
 'dtype': torch.float32,
 'eps': 1e-05,
 'experts_per_token': None,
 'final_rms': True,
 'from_checkpoint': False,
 'gated_mlp': True,
 'init_mode': 'gpt2',
 'init_weights': False,
 'initializer_range': 0.011180339887498949,
 'load_in_4bit': False,
 'model_name': 'Llama-2-13b-chat-hf',
 'n_ctx': 4096,
 'n_devices': 1,
 'n_heads': 40,
 'n_key_value_heads': None,
 'n_layers': 40,
 'n_params': 12687769600,
 'normalization_type': 'RMSPre',
 'num_experts': None,
 'original_architecture': 'LlamaForCausalLM',
 'output_logits_soft_cap': -1.0,
 'parallel_attn_mlp': False,
 'positional_embedding_type': 'rotary',
 'post_embedding_ln': False,
 'relative_attention_max_distance': None,
 'relative_attention_num_buckets': None,
 'rotary_adjacent_pairs': False,
 'rotary_base': 10000,
 'rotary_dim': 128,
 'scale_attn_by_inverse_layer_idx': False,
 'seed': None,
 'tie_word_embeddings': False,
 'tokenizer_name': 'meta-llama/Llama-2-13b-chat-hf',
 'tokenizer_prepends_bos': True,
 'trust_remote_code': True,
 'ungroup_grouped_query_attention': False,
 'use_NTK_by_parts_rope': False,
 'use_attn_in': False,
 'use_attn_result': False,
 'use_attn_scale': True,
 'use_hook_mlp_in': False,
 'use_hook_tokens': False,
 'use_local_attn': False,
 'use_normalization_before_and_after': False,
 'use_split_qkv_input': False,
 'window_size': None}
Loading reading probe for layer 30...
Loaded probe from /teamspace/studios/this_studio/mech_interp_exploration/src/probe_checkpoints/reading_probe/probe_at_layer_30.pth
Probe loaded successfully for layer 30
Starting probe testing...
Processing test cases: 100%|██████████████████████████████████████████████████| 2002/2002 [11:47<00:00,  2.83it/s]
Completed testing on 2002 cases
Overall accuracy: 0.6209

================================================================================
EVALUATION METRICS
================================================================================
Overall Accuracy: 0.6209
Classes present in data: ['very_happy', 'happy', 'slightly_positive', 'neutral', 'slightly_negative', 'sad', 'very_sad']
Number of unique classes: 7
Weighted Precision: 0.6828
Weighted Recall: 0.6209
Weighted F1-Score: 0.6239

Per-class metrics:
very_happy          : Precision=0.848, Recall=0.720, F1=0.779
happy               : Precision=0.410, Recall=0.745, F1=0.529
slightly_positive   : Precision=0.522, Recall=0.339, F1=0.411
neutral             : Precision=0.873, Recall=0.479, F1=0.619
slightly_negative   : Precision=0.569, Recall=0.734, F1=0.641
sad                 : Precision=0.611, Recall=0.759, F1=0.677
very_sad            : Precision=0.948, Recall=0.570, F1=0.712
Creating visualizations in probe_test_results...
All visualizations saved to probe_test_results/
Detailed results saved to probe_test_results/detailed_results.csv
Summary statistics saved to probe_test_results/summary_statistics.csv

================================================================================
TESTING COMPLETED SUCCESSFULLY!
================================================================================
Results saved to: probe_test_results/
Overall accuracy: 0.6209
Check the generated graphs for detailed analysis