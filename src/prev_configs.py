####
# updated-1 --> fine-tuned-dse1
# updated-2 --> common
# updated-3 --> fine-tune-dse2
# updated-4 --> fine-tune-dse3
# updated-5 --> fine-tune-dse4
# updated-old-5 --> fine-tune-oldspeed-fromv1-todse4
# updated-new-5 --> fine-tune-todse4-fromv1
# updated-new-6 --> fine-tune-todse5-fromv1
# updated-new-5-norm-util --> fine-tune-todse5-fromv1-norm-util
# updated-yizhou-5 --> fine-tune-dse4-fromv1-twosetp-yizhou
# updated-new-6 --> fine-tune-todse5-fromv1 with results of new-5
# updated-freeze4-5 --> fine-tune-todse4-fromv1-freeze4
# updated-freeze5-5 --> fine-tune-todse4-fromv1-freeze5
# updated-onlynew-tuneall-5 --> fine-tune-dse4-only-new-points-tune-all
####


## GAE
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-all-dac-tile-regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'
## GAE pragma dim
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-tile-regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'
## GAE pragma dim and extended graph block
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-tile-extended-pseudo-block-regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'
## GAE pragma dim and extended graph block connected
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-tile-extended-pseudo-block-connected-regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'
## GAE - correct edge ID - uniform reduction
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-tile-correct-edge-ID--regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'
## GAE pragma dim and extended graph block connected - block IDs saved
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-tile-correct-edge-ID-extended-pseudo-block-connected-regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'
## GAE pragma dim and extended graph block connected - block IDs saved - db of extended-graph DSE added - Aug 14th 2022
# encoder_path = '/share/atefehSZ/RL/original-software-gnn/software-gnn/save/programl/with-updated-all-data-tile-extended-graph-db-extended-pseudo-block-connected-regression_with-invalid_False-normalization_speedup-log2_no_pragma_False_tag_whole-machsuite-poly_perfutil-DSPutil-BRAMutil-LUTutil-FF/encoders.klepto'