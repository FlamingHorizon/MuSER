MUSER: MUltimodal Stress Detection using Emotion Recognition as an Auxiliary Task
=================================================================================
Yiqun Yao (*), Michalis Papakostas (*), Mihai Burzo (^), Mohamed Abouelenien (+), and Rada Mihalcea(*)

(*) Computer Science and Engineering, University of Michigan

(^) Mechanical Engineering, University of Michigan

(+) Computer and Information Science, University of Michigan

{yaoyq,mpapakos,mburzo,zmohamed,mihalcea}@umich.edu

1. Introduction

This is the code for our multimodal stress detection method using emotion recognition (activation and valence prediction) as auxiliary tasks.

We propose a late-fusion model based on textual Transformer and audio MLP. The multi-task training is handled by a speed-based dynamic sampling strategy.

===============================================================================
2. Pre-processed Data

- all_data_joint_fixed_single_task.pkl: MuSE data with sentences tokenized by Bert Tokenizer, and audio features extract with OpenSmile and eGeMaps configuration.
- all_data_omg_ready.pkl: OMGEmotions data with the same pre-processing as MuSE.

===============================================================================
3. Training a MUSER Model

Our implementation are dependent on Python3, Pytorch and Huggingface Transformers packages.

For training and testing, just run python {script-name}.py without specific arguments:

- late_fusion_joint.py: single-task learning with a late_fusion model. For stress detection, use TRIconfig.mode='classify' in line 134, else TRIconfig.mode='regression'. Also select the task labels with line 117-122.
- late_fusion_multi_task.py: multi-task learning on MuSE dataset with uniform sampling.
- late_fusion_multi_task_dynamic.py: multi-task learning on MuSE dataset with dynamic sampling.
- late_fusion_pre_fine.py: pre-training on emotion recognition tasks and fine-tune on the stress detection task. Adjust the first and second auxiliary tasks with line 198 and 210-215.
- multi_task_dynamic_omg.py: multi-task learning with OMGEmotions as an external auxiliary task, using dynamic sampling.

====================================================================================
5. Acknowledgements

This material is based in part on work supported by the Toyota Research Institute, the Precision Health initiative at the University of Michigan, the National Science Foundation (grant \#1815291), and the John Templeton Foundation (grant \#61156). Any opinions, findings, conclusions, or recommendations in this material are those of the authors and do not necessarily reflect the views of the Toyota Research Institute, Precision Health initiative, the National Science Foundation, or the John Templeton Foundation. 