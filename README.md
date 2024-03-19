# Training language models to follow instructions with human feedback

# Overview (Kit)


# Architecture Overview (Tony)

## Methodology Overview

<p align="center">
  <img src="Images/LLM-HF-figure.JPG" alt="Methodology Overview">
</p>

* **Goal**: Build a model environment to align models to be more helpful, honest, and harmless.
* Step 1 (Supervised Fine-Tuning [SFT]): Using demonstration data, which includes prompts and expected outputs, to fine-tune a GPT model to learn directly from example.
* Step 2 (Reward Modeling [RM]): Using comparison data, which includes prompts and generated outputs with rankings, to select outputs that are more aligned to human preference.
* Step 3 (Reinforcement Learning from Human Feedback [RL or PPO]): Language model (InstructGPT) is further fine-tuned using reinfocement learning, guided by reward model to generate ouputs that maximize the predicted human preference. 


## Data Collection

* Prompts primarily consist of OpenAI prompts from actual users that agreed to use a GPT model on a playground interface.
* In addition, labelers wrote prompts themselves for an initial source of instruction-like prompts to bootstrap the process. This include:
    * Plain: Labelers were asked to come up with an arbitrary task,
    * Few-Shot: labelers came up with an instruction and multiple responses to choose from, and
    * User-based: lablers came up with prompts that were use-cases to users on a waitlist to increase diversity in our data.
* Prompts include the following categories:
    * Brainstorming
    * Classification
    * Extract
    * Generation
    * Rewrite
    * Chat
    * Closed QA
    * Open QA
    * Summarization.

<p align="center">
  <img src="Images/dataset_sizes.JPG" alt="Dataset Sizes Overview">
</p>

* 13k training prompts on Supervised Fine-Tuning.
* 33k training prompts on Reward Modeling.
* 31k training prompts on Reinforcement Learning.


## Step 1: Supervised Fine-Tuning (SFT)

* Train on Demonstration Data: Prompts with corresponding labeled outputs from labelers (to handle quality control).
* Model Purpose: Training the language model to replicate and generate responses, improving its ability to generate text that meet the alignment standards
* Model Objective: Minimize the difference between modeling generated outputs and human-provided target outputs

---

**Algorithm 1:**
$\Theta_{ft} \leftarrow \text{SFT}(X, Y, \Theta)$

#### Inputs
- `X, Y`: Labeled demonstration data, where `X` represents the input prompts and `Y` represents the corresponding target outputs.
- `Θ`: Initial parameters of the GPT-3 model.

#### Output
- `Θ_ft`: The model parameters adjusted through the fine-tuning process.

#### Hyperparameters
- `N_epochs ∈ ℕ`: Number of epochs, representing the total number of passes over the entire training dataset.
- `batch size`: The number of training examples utilized in one iteration.
- `η_init`: Initial learning rate, a scalar used to adjust the magnitude of parameter updates at the start of training.
- `β1, β2, ε`: Hyperparameters specific to the Adam optimizer. `β1` and `β2` control the exponential decay rates for the moment estimates. `ε` is a small scalar added to prevent division by zero in the optimizer's calculations.

#### Process
1. **Initialization**: Prepare the GPT-3 model with the pretrained weights `Θ` and initialize the Adam optimizer with the specified initial learning rate `η_init` and hyperparameters.

2. **Training Loop**:
   - For each epoch `i` from 1 to `N_epochs`:
     - **Cosine Decay Adjustment**: Adjust the learning rate `η` for the current epoch using the cosine decay formula. The adjusted learning rate `η` is calculated as follows:
       ```
       η = η_init * (1 + cos(pi * i / N_epochs)) / 2
       ```
       This formula gradually decreases the learning rate from `η_init` to near 0 following a cosine curve over the epochs.
     - For each batch `(input_batch, target_batch)` in the dataset:
       - **Forward Pass**: Compute the predicted output `Y_hat` for `input_batch` using the GPT-3 model.
       - **Loss Calculation**: Compute the loss using a suitable `LossFunction` that measures the difference between `Y_hat` and `target_batch`.
       - **Backward Pass and Parameter Update**: Use the Adam optimizer with the current learning rate `η` to update the model parameters `Θ_ft` based on the gradients of the loss function.

3. **Output**: Return the fine-tuned model parameters `Θ_ft`.

---


## Step 2: Reward Modeling (RM)

* Train on Comparison Data: Rather than training a model on generating text, we want the model to select the best output that the SFT model generated. Thus, the data used to train the reward model includes a prompt along with rankings of desired outputs

<p align="center">
  <img src="Images/RM_data.JPG" alt="RM Data">
</p>

* Model Purpose: Evaluate the quality of generated texts providing a reward signal that indicates how well a piece of text meets the alighment criteria. This acts as a proxy for human judgement
* Model Objective: accuratley predict human preferences among different generated outputs from the same input, using the following loss function:

$$\text{loss}(\Theta) = -\frac{1}{\binom{K}{2}}E_{(x,y_w,y_l)~D}[log(\sigma(r_\Theta(x,y_w)-r_\Theta(x,y_l)))]$$ 

---

**Algorithm 2**
$\Theta_{RM} \leftarrow \text{RM}(X, Y_K, \Theta)$

#### Inputs
- `D`: Dataset of sets, each containing `K` responses \((x, \{y_1, y_2, ..., y_K\})\), where `x` is an input prompt and `\{y_1, y_2, ..., y_K\}` are the responses with implied pairwise preferences among them based on human judgments.
- `Θ_initial`: Initial parameters for the reward model

#### Output
- `Θ_RM`: Adjusted reward model parameters after the training process.

#### Hyperparameters
- `N_epochs ∈ ℕ`: Number of training epochs, indicating the total number of complete passes over the dataset `D`.
- `batch size`: The number of sets processed in one iteration of training.
- `η`: Learning rate, a scalar used to adjust the magnitude of parameter updates.
- `Optimizer`: Adam, including its specific hyperparameters `β1`, `β2`, and `ε`.

#### Process
1. **Initialization**: 
   - Initialize the reward model with parameters `Θ_initial`.
   - Prepare the optimizer with the learning rate `η` and its hyperparameters.

2. **Training Loop**:
   - For `epoch` in range(1, `N_epochs`+1):
     - Shuffle dataset `D` to ensure a random distribution of data in each epoch.
     - For each set \((x, \{y_1, y_2, ..., y_K\})\) in `D`:
       - Initialize `set_loss` to 0.
       - Generate all possible pairs `(y_w, y_l)` from the set of `K` responses, considering `y_w` is preferred over `y_l`.
       - For each pair `(y_w, y_l)` in the set:
         - Compute `score_diff = r_Θ(x, y_w) - r_Θ(x, y_l)`.
         - Calculate the logistic loss for the pair: `pair_loss = -log(sigma(score_diff))`.
         - Accumulate the loss: `set_loss += pair_loss`.
       - Normalize set loss by the total number of pairs $\binom{K}{2}$ in the set to compute the average loss: $averageLoss = \frac{setLoss}{\binom{K}{2}}$.
       - Update `Θ_RM` using the optimizer to minimize `average_loss`.

3. **Output**: 
   - Return the trained reward model parameters `Θ_RM`.

---

## Step 3: Reinforcement Learning




# Critical Analysis (Kit)


# Resources
**Original Devlopement of Reinforcement Learning from Human Feedback**
Christiano, P. F., Leike, J., Brown, T., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, pages 4299–4307

**Previous Work on Training Language Models to Follow Instructions**
Yi, S., Goel, R., Khatri, C., Cervone, A., Chung, T., Hedayatnia, B., Venkatesh, A., Gabriel, R., and Hakkani-Tur, D. (2019). Towards coherent and engaging spoken dialog response generation using automatic conversation evaluators. arXiv preprint arXiv:1904.13015

**Overview of Risks with Langauge Models**
Bender, E. M., Gebru, T., McMillan-Major, A., and Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623.

**Fine-tuning Langauge Models to Improve Model's Ability to Adhere to Better Values on a Question Answering Task**
Solaiman, I. and Dennison, C. (2021). Process for adapting language models to society (palms) with values-targeted datasets. arXiv preprint arXiv:2106.10328.

**A Promising Future Path for RLHF Using Control Codes**
Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., and Socher, R. (2019). Ctrl: A conditionaltransformer language model for controllable generation. arXiv preprint arXiv:1909.05858.

**Proposal to Align Superhuman Systems using RLHF**
Leike, J., Krueger, D., Everitt, T., Martic, M., Maini, V., and Legg, S. (2018). Scalable agent alignment via reward modeling: a research direction. arXiv preprint arXiv:1811.07871.

**Theoretical Research on Alignment**
Soares, N., Fallenstein, B., Armstrong, S., and Yudkowsky, E. (2015). Corrigibility. In Workshops at the Twenty-Ninth AAAI Conference on Artificial Intelligence.

**A Promising Future Path for RLHF Modifying the Sampling Procedure**
Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., and Liu, R. (2019). Plug and play language models: A simple approach to controlled text generation. arXiv preprintarXiv:1912.02164.

# Paper Citation
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. [Preprint]. arXiv. https://arxiv.org/abs/2203.02155
