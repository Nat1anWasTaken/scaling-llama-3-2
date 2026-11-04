<!--Copyright © 2026 Mu-Tien Kuo-->

# Description

Experiments at small scales allow us to explore training methods without committing the compute required for a full pretraining run. Yet a method that performs well on a small model may behave differently as model size and training duration increase. Scaling laws [1] provide one approach to predicting these differences, while careful experiment design allows us to test whether our findings transfer.

In this lab, you are to develop a training recipe through a series of smaller experiments, then use it to train a 1B model from scratch on 200B tokens of FineWeb-Edu [2].

Your goal is to get the **lowest** perplexity on our evaluation stack after training on exactly 200B tokens, with a context window of 4096 tokens. The token budget counts non-padding tokens processed during training, including repeated tokens each time they are processed.

You are encouraged to explore optimizers, initialization, parameterization, learning rates, schedules, batch sizes, and other methods that improve training. Sweeps are acceptable, but you are encouraged to find a recipe through small-scale sweeps that successfully transfers to the final run. Nonetheless, you should consider what each experiment tells you about your eventual training recipe, since performing well at a smaller scale does not necessarily imply performing well at the final target.

## Model Architecture

Your models should follow the architectural features of Llama 3.2, including causal decoder blocks, RMSNorm, SwiGLU, rotary position embeddings, grouped-query attention, and tied input and output embeddings.

You may experiment with model dimensions, attention head dimensions, query-to-KV head ratios, and attention scaling. The configurations below are provided for reference, rather than as required configurations. You may also train models at intermediate scales.

| Requested budget | Layers (L) | Hidden (d) | MLP (m) | Query heads | KV heads | Actual parameters |
| --- | --- | --- | --- | --- | --- | --- |
| 150M | 6 | 768 | 3,072 | 12 | 3 | 149,825,280 |
| 300M | 9 | 1,024 | 5,184 | 16 | 4 | 298,273,792 |
| 500M | 11 | 1,536 | 4,672 | 24 | 6 | 498,731,520 |
| 750M | 13 | 1,792 | 5,888 | 28 | 7 | 745,749,760 |
| 1B | 15 | 1,792 | 8,000 | 28 | 7 | 995,432,704 |

The parameter counts above assume a vocabulary of 128,256 tokens, tied input and output embeddings, bias-free projections, and two RMSNorms per decoder layer plus a final RMSNorm.

- Recommendations towards intermediate scales
    
    In case you want to explore different compute optimal rations, you may refer to the following section on a reference to how Llama’s architecture works 
    
    ### Decoder Layer Forward Pass
    
    Each decoder layer consists of an attention sublayer followed by an MLP sublayer. Both use pre-normalization and a residual connection:
    
    $$
    r=x+\operatorname{Attention}\bigl(\operatorname{RMSNorm}_{\mathrm{attn}}(x)\bigr)
    $$
    
    $$
    y=r+\operatorname{MLP}\bigl(\operatorname{RMSNorm}_{\mathrm{mlp}}(r)\bigr)
    $$
    
    The two RMSNorms have separate learned scale vectors. Both residual additions preserve the input shape ([B,T,d]), where (B) is batch size and (T) is sequence length. See torchtune’s decoder layer for a reference.
    
    For attention, first project the normalized input (u) into queries, keys, and values:
    
    $$
    Q=\operatorname{RoPE}(uW_Q),\qquad
    K=\operatorname{RoPE}(uW_K),\qquad
    V=uW_V.
    $$
    
    After reshaping into heads, each query head attends to its assigned KV head:
    
    $$
    A_i=
    \operatorname{softmax}
    \left(sQ_iK_{g(i)}^\top+M\right)V_{g(i)}.
    $$
    
    Here, (g(i)) maps a query head to its KV head, (M) is the causal attention mask, and (s) is the attention scaling factor. A conventional starting point is ($s=1/\sqrt{d_h}$), where ($d_h$) is the head dimension. Concatenate the query-head outputs and apply the output projection to return to width (d). Attention reference
    
    For the MLP, use SwiGLU:
    
    $$
    \left(
    \operatorname{SiLU}(zW_{\mathrm{gate}})
    \odot
    zW_{\mathrm{up}}
    \right)W_{\mathrm{down}}
    $$
    
    The gate and up projections independently map ($d\rightarrow m$), their outputs are multiplied elementwise, and the down projection maps ($m\rightarrow d$). These equations use row-vector notation. SwiGLU reference
    
    After the last decoder layer, apply a final RMSNorm and project to vocabulary logits using the tied embedding matrix.
    
    ### Choosing Intermediate Dimensions
    
    The MLP intermediate dimension (m) controls the width of both the gate and up projections. You are free to choose it independently of the attention dimensions.
    
    Two useful starting points are:
    
    - **($m\approx\frac83d$):** A common SwiGLU choice that approximately matches the matrix parameter count of a conventional two-projection MLP with width (4d). Meta’s Llama implementation applies this reduction before optional multipliers and rounding. Implementation reference
    - **($m=4d$):** The ratio used by Llama 3.2 1B, which has hidden dimension 2048 and intermediate dimension 8192. Configuration reference
    
    You may round your chosen intermediate dimension to a convenient multiple, such as 64, 128, or 256:
    
    $$
    m=a\left\lceil\frac{\rho d}{a}\right\rceil,
    $$
    
    where ($\rho$) is your chosen expansion ratio and (a) is the rounding multiple. For example, with (d=1024), ($\rho=8/3$), and (a=64), this gives (m=2752). This is a starting heuristic, rather than a required formula.
    
    For attention dimensions, the usual relationship is:
    
    $$
    d_h=\frac{d}{h_q}
    \qquad
    d_{\mathrm{KV}}=h_{kv}d_h
    $$
    
    where ($h_q$) and ($h_{kv}$) are the query and KV head counts. Under this convention, choose (d) divisible by ($h_q$), and ($h_q$) divisible by ($h_{kv}$). Standard RoPE also requires an even head dimension.
    
    Before reshaping into heads, queries have width (d), while keys and values each have width ($d_{\mathrm{KV}}$). Please check that your chosen attention kernel supports the resulting head dimension.
    

## Training Stack

You are expected to develop a persistent training stack that is mostly handwritten. You may fork Flock’s (extremely naive) training stack as a starting point.

The intention is for this stack to grow alongside your experiments and remain useful in subsequent labs. You should become familiar with how your model, optimizer, data loading, packing, loss calculation, and checkpointing are implemented, since these details affect both the results of your experiments and your ability to understand them.

You may use PyTorch and existing libraries for components such as attention kernels and distributed training. Nonetheless, you are expected to understand and maintain the training logic you use. Please make sure both members of your pair can explain how your most promising methods are implemented and why you expect them to work.

## Compute Grants

To encourage experiment planning, each *pair* will receive compute grants in a ladder-like fashion. Each grant permits one run using **up to** its assigned H200-hour allowance.

The grant levels are based on the compute required by a well-optimized PyTorch trainer using `torch.compile`, FlashAttention-3, and sequence packing, training reference models at 200 tokens per parameter. The allowances are intended to give pairs some breathing room for differences in implementation.

H200-hours count compute across all GPUs used in a run. For example, using 8 H200s for 2 hours consumes 16 H200-hours.

| Grant allowance per run | Ref scale | Ref tokens | #grants | Availability |
| --- | --- | --- | --- | --- |
| 40 H200-hours | 150M | 30B | ~60 | Throughout both weeks |
| 110 H200-hours | 300M | 60B | 8 | Throughout both weeks |
| 266 H200-hours | 500M | 100B | 3 | 1 before calibration, 4 additional after |
| 540 H200-hours | 750M | 150B | 1 | Must launch before 11/12 23:59 |

The reference scales describe the intended size of each compute allowance. They do not restrict your experiments to those parameter counts. You may choose other model sizes using the architecture scaling equation.

You may use a grant for a smaller experiment or stop a run early. Nonetheless, each launched experiment consumes a grant at its assigned level. These grants are intended to provide a certain number of experimental opportunities, rather than a compute wallet that can be divided into unlimited replacement runs. Resuming an interrupted run from its checkpoint is considered part of the same run.

You are free to choose the model sizes and training horizons of your exploratory experiments within their grant allowances. The required 750M calibration run should use the scaling equation and train for 200 tokens per parameter.

In addition to these exploration grants, each pair is allocated one final 1B run on 200B tokens. This run has a fixed model configuration, derived from the scaling equation, and a fixed token endpoint. Improving your training stack may allow you to finish sooner, but does not permit training on additional tokens.

## Evaluation

You will be evaluated on perplexity based on the following three datasets

- fineweb_edu:  https://huggingface.co/datasets/cerulean-works/fineweb-edu-dev
- base_easy / base_easy:bpb: OLMo 3’s base easy suite
- paloma: 309 domain BPB, for domain breakdown

<aside>
⚠️

You should exclude the 5k fineweb-edu eval samples from your training run

</aside>

You should use Flock’s evaluation module to run your scoring https://docs.cerulean.works/flock/main/guide/evals/run-evaluation/. Running the three eval commands looks like this:

```bash
flock eval run --task fineweb_edu --model <repo@rev> --dtype float32 --out <dir>
flock eval run --suite base_easy --model <repo@rev> --dtype float32 --out <dir>
flock eval run --suite paloma --model <repo@rev> --dtype float32 --out <dir>
```

During the timeframe of this lab, flock will gradually mature in its model registry and eval capabilities, yet before this time, please be forgiving for the many faults our current system may have. You are more than welcome to open issues on GitHub or Slack! 

## Experiment Planning and Calibration

You are required to launch a 750M calibration run before the 11/12 24:00. You are recommended to launch sufficiently early that its results can inform your W2 experiments.

The 266 H200-hour grant available during W1 gives you an opportunity to test whether your findings transfer before committing to 750M. During W2, the additional grants may be used to investigate your calibration results, compare candidate recipes, or test modifications before selecting your final approach.

You are not required to use the same recipe throughout the lab. Changing your approach in response to an experiment is completely acceptable. Please record what changed, why you changed it, and which results support the revised recipe.

In particular, please remember that learning rates depend on the optimizer, parameterization, batch size, and training duration. There is no prescribed peak learning rate for this lab. You should instead consider how your update rules and hyperparameters transfer as the model and training horizon change.

Pairs are expected to maintain a research log recording their hypotheses, experiments, and findings. Please include why you chose an experiment and what you learned from it, rather than only recording its configuration and final PPL. A failed transfer experiment can still be useful if it helps explain a limitation or changes your subsequent decisions.

## Forecasting

In addition to the final model score, this lab will recognize pairs that accurately predict their training results.

Before launching your required 750M calibration run and your final 1B run, write a set of the following for each benchmark in your experiment logs:

- Your model and training configuration, including the intended token endpoint.
- Your estimated endpoint perplexity on the designated evaluation set.
- An 80% prediction interval, representing a range you believe has an 80% chance of containing the result.
- A short explanation of your estimate and the experiments supporting it.

You may use scaling-law fits, comparisons against a baseline, or another approach you find reasonable. There is no required fitting method or number of calibration experiments. Deciding how much experimentation to spend improving your prediction is part of planning your runs.

Your forecast must be submitted before the corresponding run begins and applies to the submitted recipe. You may revise your recipe between the 750M calibration and the final run, but you may not revise an earlier forecast after seeing its outcome.

Forecast error will be calculated in log-perplexity space:

$$
E=\left|\ln(\widehat{\mathrm{PPL}})-\ln(\mathrm{PPL}_{\mathrm{actual}})\right|.
$$

The forecasting score is the average error across the two designated runs, where **lower is better**. Prediction intervals will be recorded for discussion but will not contribute to the score.

You may also post updated predictions while a run is training. These updates can help explain how your expectations changed, but do not replace the original scored forecast.

Forecasting is scored separately from final model quality. A pair may win the main lab through a strong recipe even if its predictions were inaccurate.

## Leaderboards and SOTA Points

The main leaderboard will rank pairs by the final 1B model’s perplexity after 200B training tokens. A separate forecasting leaderboard will rank pairs by their forecast error.

Intermediate experiment results will be shared, but there will be no competitive PPL leaderboard or SOTA points for individual smaller scales. The purpose of these experiments is to help you choose a recipe that performs well at the final target, and you should feel free to run an experiment that answers a useful question even if it does not produce the best small-model PPL.

SOTA points will be awarded as a function of relative performance in the final training runs. Since I have no idea how our scores will turn out, I’m withholding the final SOTA point calculation until our calibration run results come out. Additionally, SOTA points may be retroactively adjusted by the comprehensiveness of experiment logs, so please keep a clean record of what you’ve done. 

## Logging and Submission

You are expected to log your training metrics to wandb under the `lab6-predictable-scaling` project. To facilitate comparisons across peers, you are required to log the following metrics:

- `train/loss`: Average loss over the duration of `logging_steps`.
- `train/grad_norm`: Gradient norm before clipping.
- `train/learning_rate`: Log each parameter group separately if they use different learning rates.
- `train/tokens_per_second`: Training throughput at the logging step.
- `train/total_tokens_seen`: Cumulative non-padding training tokens processed.
- `eval/loss`: Token-averaged cross-entropy on the designated development set.
- `eval/perplexity`: Perplexity calculated from that evaluation loss.

Evaluation should be performed no less frequently than once every 10% of the intended training steps. Include your model configuration, initialization, optimizer settings, batch size, schedule, and training duration in the wandb configuration.

Before launching the final run, commit your chosen recipe and submit its configuration and forecast. Each pair is allocated one final training attempt; the checkpoint at the prescribed 200B-token endpoint will determine its main leaderboard score.

Upon completion, upload your model weights and tokenizer to HuggingFace and set `eval_model_id` in your OJ script. Include your training configuration, wandb run link, forecast submission, and research log so that the recipe, token budget, and training history can be verified.

## Deadlines

- Before the 2026/11/12 23:59, submit your forecast and launch the required 750M calibration run.
- Before the end of W2, submit your final recipe and forecast, and launch your 1B/200B run.
- Calendar dates and final-run completion arrangements will be announced before the lab begins.

## References

[1] Training Compute-Optimal Large Language Models
[2] FineWeb-Edu on Flock
[3] Llama 3.2 1B
[4] Llama 3.2 reference implementation in torchtune
[5] The Llama 3 Herd of Models
