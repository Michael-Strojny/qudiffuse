Under review as a conference paper at ICLR 2025
 000
001
002
003
004 Paper under double-blind review 005
LATENT DIFFUSION WITH LLMS FOR REASONING
Anonymous authors
006
007
008
009
010
011
012
013
014 latent diffusion models with an encoder-decoder transformer architecture provides
015 a scalable way to address some of the fundamental shortcomings posed by autore-
1 INTRODUCTION
ABSTRACT
Despite the widespread adoption of large language models with hundreds of bil- lions of parameters, these models still struggle on complex reasoning benchmarks. In this paper, we argue that the autoregressive nature of current language models are not suited for reasoning due to fundamental limitations, and that reasoning requires slow accumulation of knowledge through time. We show that combining
gressive models. Diffusion models can arrive at predictions through many forward passes in latent space, and their reasoning is not handicapped by the order of the tokens in the dataset. Through our experiments, we show that latent diffusion lan- guage models is a feasible approach towards scalable language models that have general complex reasoning abilities.
016
017
018
019
020
021
022
023
024
025
026
027 where language models elicit impressive emergent capabilities. However, even the biggest corporate
028 LLMs still struggle with complex reasoning benchmarks (Sawada et al., 2023). Prior work has
029 shown that LLMs are limited by their autoregressive nature because the FLOPs used to generate
053
task, the required semantics might not be representable by its token embeddings (e.g. spatial rea- soning). LDMs also do not suffer from memory requirements and instabilities encountered by back-
In recent years, autoregressive large language models (LLMs) have become the de-facto for natural language generation (Team et al., 2024; Radford et al., 2019). The excellent scalability of transformers combined with the availability of large datasets has led to many practical applications
each token is constant regardless of the difficulty of the token (Bachmann & Nagarajan, 2024). Ad- ditionally, the model has to generate tokens in the order of the dataset it is trained on and therefore cannot solve easier subproblems first (Bachmann & Nagarajan, 2024). The model will not ex- plicitly generate easier reasoning chains first unless explicitly fine-tuned to do so on a subset of tasks.
030
031
032
033
034
035 Numerous approaches have tried to tackle this problem such as chain-of-thought (CoT) prompting
(Wei et al., 2022), enhancing model reasoning with longer context (Dai et al., 2019), or encoding recurrence into transformers (Hutchins et al., 2022; Bulatov et al., 2022). These approaches, however, are not a general solution to these shortcomings because pretraining datasets are rarely CoT prompted, and compute allocated to each token is still constant. This is a fundamental limitation when solving math problems. For example, not all tokens have equal difficulty, and often times the answer to easier subproblems lead to better answers for harder subproblems (e.g. geometry, algebra). Even though recurrent models can perform many forward passes in latent space, prior work has not been able to scale efficiently due to its memory requirements, and it has been observed that long unrolls lead to exploding or vanishing gradients (Vicol et al., 2021).
036
037
038
039
040
041
042
043
044
045
046
047
048
049
050 self-correction abilities, where reasoning steps can be generated non sequentially and that the model
051 learns to correct its wrong answers from easier (and correct) reasoning steps (Ye et al., 2024). LDMs
052 perform reasoning in latent space which is semantically richer than discrete tokens. For a specific
In this paper, we propose to combine latent diffusion models (LDMs) with encoder-decoder trans- formers in an attempt to solve the mentioned shortcomings that are posed by autoregressive LLMs. In contrast to traditional LLMs, LDMs generate a latent vector by iteratively denoising Gaussian noise throughout many timesteps, which intuitvely makes it more suitable for tasks that require ex- trapolating many facts over long horizons. Prior work has shown that text diffusion models elicit
1
Under review as a conference paper at ICLR 2025
 054
055
056
057
058 diffusion text models, strengthening our claim that operating in latent space yields improvements
059 over operating with discrete tokens (He et al., 2022; Lovelace et al., 2024b).
propagation through time (where it is a common practice to set max timesteps to 1000 or 4000). This is due to the fact that gradients are not propagated through the same parameters multiple times (Ho et al., 2020; Nichol & Dhariwal, 2021) which makes it an appealing candidate to solve the aforemen- tioned shortcomings. It has also been shown that latent diffusion text models outperform discrete
060
061
062
063
064
065
066
067
068
069
070
071
072
073
074
075
076
077
078 By iteratively denoising an image distribution from Gaussian noise, diffusion models have been
079 able to outperform generative adversarial networks on image generation benchmarks. Continuing
080 research on LDMs have also found that these models generate more diverse image samples, and
techniques such as Min-SNR-γ (Hang et al., 2023) and progressive distillation (Salimans & Ho, 2022) improved the efficiency of the training and inference such that LDMs can now generate high quality images and videos at a fraction of the cost (Rombach et al., 2022). Since this paper combines diffusion with autoregressive encoder-decoder language models, we briefly review the literature on the reasoning abilities of LLMs and some basic concepts to understand diffusion models.
We summarize the benefits of combining LDMs with encoder-decoder language models for complex reasoning task as follows:
2
RELATED WORK / PRELIMINARIES
1. It can do reasoning in semantic space and does not rely on discrete tokens where the accu- mulation of knowledge per forward pass only amounts to that particular generated token.
2. It can perform reasoning non-sequentially regardless of the order of the tokens in the train- ing data. Throughout denoising steps, LDMs elicit self-correction where correct reasoning steps lead to corrections on harder reasoning steps.
3. It does not run into memory bottlenecks and instabilities that are encountered by recur- rent transformers as we scale to larger unroll lengths because gradients are not propagated through the same parameters multiple times.
Generative pretrained transformers (GPTs) have significantly transformed natural language process- ing demonstrating exceptional scalability and achieving state-of-the-art performance on a variety of downstream tasks, including translation, summarization, and instruction following (Achiam et al., 2023). Meanwhile, image generation also had a renaissance powered by LDMs (Yang et al., 2023).
081
082
083
084
085
086
087
088
089
090
091 image (by adding Gaussian noise), can be formulated as xt = √α ̄tx0 + √1 − α ̄tε where we sample
092 noise ε ∼ N(0,I), and α ̄t is a noise scheduling hyperparameter that controls how noise is applied
093 on different timesteps. In order to learn the reverse process to reconstruct the target image x0, a
094
model θ is learned to predict the noise at each timestep, which is optimized by minimizing a simple mean squared error loss between εˆ = εθ(xt,t), the estimated noise of time t, and ε: Lsimple (θ) =
2.1 DENOISING DIFFUSION PROBABILISTIC MODELS
DDPMs (Ho et al., 2020) are a class of diffusion models that iteratively construct an image from random Gaussian noise. We define x0 as the original image which is slowly corrupted into random Gaussian noise iteratively. The forward process, which converts the original image into a corrupted
 095
096 ∥εθ (xt , t) − ε∥2 . During each iteration of inference, random Gaussian noise can then be turned into
an image according to a target data distribution by iteratively removing εθ (xt , t). For each sampling
097 
1 1−αt
098 step, the denoised image for the next step is given by xt−1 = √αt xt − √1−α ̄t εθ (xt, t) + σtz
099 where we sample noise z ∼ N (0, I), σt is the standard deviation, and α ̄t = Qts=1 αs.
100
  101 102 103
104 ditioning information for diffusion (Peebles & Xie, 2023). Conditional image generation can be
105 formulated as the probability of an image x given information such as a class label c, pθ(x|c),
106 where c is an additional information (such as the class of an image). The DiT architecture con-
107
sists of adaLN-zero blocks (Peebles & Xie, 2023) which incorporate conditioning information by regressing dimension-wise over the scale and shift parameters used in adaptive layer normalization
2.2 SCALABLE DIFFUSION MODELS WITH TRANSFORMERS
Diffusion transformers (DiT) is a variant of transformer that has been modified to incorporate con-
2

Under review as a conference paper at ICLR 2025
 108
109
110
111
112 which leads to faster convergence empirically. DiT has shown to outperform traditional U-Net as
113 backbone for diffusion due to its remarkable scaling properties.
(adaLN) from the sum of the embedding vectors of the current timestep t and class c. In addition to adaLN, DiT also regresses dimension-wise scaling parameters that are added prior to any residual connections. They further initialize all multilayer perceptron to output the zero-vector for α since this initializes the residual block to an identity block (by adding zero to the residual connections)
114 115 116
117 Due to the difficulty of reasoning tasks, LLMs perform poorly when they are tasked to directly
118 output an answer to a difficult reasoning problem. Therefore, CoT is a technique to improve LLM
119 accuracy by fine-tuning it to output a reasoning chain before the final answer (Wei et al., 2022). This
2.4 STEP-DPO
2.3 DIFFUSION-OF-THOUGHT
increases LLMs performance on hard reasoning benchmarks because the model can generate easier reasoning first that can aid it in finding the final answer. Diffusion-of-thought (DoT) attempts to take it a step further by having a discrete diffusion model diffuse CoT tokens. The authors found out that DoT elicits self-correction abilities which is in contrast to traditional LLMs Huang & Chang (2022). Our work attempts to take it a step further by augmenting it with LLMs so that it can get the best of both worlds (efficient pretraining and strong reasoning abilities).
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134 model architecture, we show that diffusion can be an add-on to step-DPO. 135
Mathematical reasoning is recognized as a long-chain reasoning ability in LLMs (Lai et al., 2024). Previous work has tried to tackle this by applying Direct Preference Optimization (DPO) (Rafailov et al., 2024) to the reasoning chain with the correct answer but with limited success. Step-DPO addresses this issue by applying DPO to each reasoning step, and curate a dataset that contains pairwise preference data generated by the model itself, which has been shown to improve training compared to GPT-4 generated data and human labeled data Lai et al. (2024). With our proposed
136 137 138
139 self-improving loop without the addition of annotated data (Tian et al., 2024). Since outputs from
140 MCTS are usually in much better quality, the gap ensures that LLM can continue to self-improve.
141 Results show that this method improves performance by as much as 30% for GSM8K and MATH
161
through many forward passes in the diffusion model.
2.5 INTEGRATING MONTE CARLO TREE SEARCH FOR LLM REASONING
Contuining work on LLM reasoning has turned to Monte Carlo tree search (MTCS) to create a
benchmarks Tian et al. (2024). Our proposed method can also be added on to the MTCS-based reasoning approaches.
142
143
144
145
146
147
148
149
150
151
152 decoder is fine-tuned to generate the original sequence given the encoder representation. 153
3 PROPOSED METHOD
In this paper, we propose to merge the encoder-decoder language model with LDMs in an attempt to enhance reasoning in natural language processing. We use pretrained encoder-decoder LLMs as our base model since these LLMs already contain high-level semantics that have been learned from large corpus of text. Particularly, we use BART (Lewis et al., 2019) extensively throughout our experiments to obtain its encoder representations. In constrast to next sequence prediction, the
154
155
156
157
158 and input tokens inherently have different lengths. Second, we train a diffusion model such that
159 the diffusion transformer denoises the target sequence compressed latent conditioned on the input
160 sequence compressed latent. Reasoning is achieved by iteratively constructing the target latent
The main training consists of two stages. First, we fine-tune the decoder and an autoencoder such that the variable length encoder representation can be compressed to a fixed length latent, which can then be decoded back to its original token sequence. This improves reliability and efficiency because diffusion models are more compute efficient at training smaller dimensional latent variables
3

Under review as a conference paper at ICLR 2025
 162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181 compressed input latent which is then used to condition the diffusion transformer to generate a
182 compressed target latent. This compressed representation is then passed through the autoencoder
183 and decoder to generate the predicted target sequence.
 Figure 1: Overview of our proposed architecture to improve reasoning.
During inference, the input sequence is fed into the encoder and autoencoder to obtain the
184
185
186
187
188
189
190 Formally, we define a sequence of tokens as x = (w1 , ..., wn ) which are sampled from a dataset to
191 get its inputs and corresponding targets (xinput,xtarget) ∼ D. Then, we aim to learn a language
192 autoencoder θ such that x can be reconstructed by passing through an encoder Eθ and decoder Dθ;
The following sections describe in more detail the architecture of the latent models and diffusion models and how they are trained.
3.1 LATENT MODELS FINE-TUNING
that is, x ≈ Dθ(Eθ(x)). In our setting, Eθ(x) = Eae(Elm(x)), and Dθ(x) = Dlm(Dae(x)) where both the encoder and decoder are composed by an autoencoder denoted by Eae and Dae and a pretrained BART encoder and decoder denoted by Elm and Dlm, respectively. Since changes in the dimensionality of the latent representation can lead to drastic changes in final performance (He et al., 2022; Nichol & Dhariwal, 2021), we compress the encoder representation to a fixed latent space with length lae = 16 and dimension dae = 256. The autoencoder architecture consists of only cross-attention transformer blocks where first block queries are learned from learnable hyper- parameters of the target dimensions, and key and values are learned from the encoder representation
193
194
195
196
197
198
199
200 or compressed representations. We did not ablate over the autoencoder design choice. Our goal is to
201 have the compressed latents contain both low level features and high level semantics instead of sim-
202 ply compressing the token sequences, so we freeze Elm during all training stages because high level
semantics are obtained from BART since they are not retrained to overfit on simply compressing the data.
203
204
205
206
207
208
209
210
211
212 diffusion. We follow the standard DDPM approach to train x0 and train the variance Σθ with the
213 full loss L(θ) = − log pθ(x0|x1)+ Pt DKL(q∗(xt−1|xt, x0)∥pθ(xt−1|xt)). In preliminary exper-
214 iments, we observed that predicting x0 instead of εθ was crucial to generate coherent text, and that
215
pretrained encoder-decoder transformers are not sensitive to small pertubations of encoder represen- tations. We use a cosine schedule with the max timestep as T = 1000 since higher T improves the
3.2 LATENT DIFFUSION
The second stage simply consists of learning a diffusion model in the latent space learned by the autoencoder. The compressed latents with length lae = 16 and dimension dae = 256 are then pro- jected up to dimension dproj which is then reshaped into length ldiffusion and a fixed dimension ddiffusion = 768. Since DiT scales with decreasing patch size (or increasing sequence length), we ablate sequence length for DiT to determine whether the same scaling law holds for latent text
4

263
264
265
266
267
268
269
Table 1: Single digit additions Architecture Accuracy↑
Latent Diffusion (T=500) 97.2 Latent Diffusion (T=1000) 96.7 Latent Diffusion (T=4000) 97.3
BART (First token as answer) 1.3 BART (Last token as answer) 0.3
Under review as a conference paper at ICLR 2025
 216
217
218
219
220
221
222
223 computations before outputting tokens to improve Perplexity. If we add noise to the encoder rep-
224 resentation during training, it learns to differentiate between noise and signal from representations,
log-likelihood of the generated samples (Nichol & Dhariwal, 2021), and that we can always sample more efficiently using different samplers from the literature that trade off sample quality.
3.3 IMPROVEMENTS
In addition to using diffusion to predict the target tokens, we could alternatively concatenate the input token sequences or representations to the decoder input to allow the decoder to do additional
teaching it that the diffusion output contains useful semantics but are not always reliable. Alterna- tively, we could also use encoders trained with contrastive learning to improve the quality of the latent representations. This allows the architecture to retain GPT performance while being able to solve additional reasoning tasks with diffusion output. If the latent representations from diffusion is unreliable, then the model defaults to autoregressive inference (Lovelace et al., 2024a). We opt to only give diffusion output to the decoder since the performance improvements would also depend on the decoder which would not reflect the capabilities of diffusion.
225
226
227
228
229
230
231
232
233
234
235
236
237 enhanced reasoning. Since representation is learned using traditional LLMs, the diffusion model is
238 able to directly utilize high-level semantics without the inefficiencies of training diffusion. Addi-
3.4 COMPARISON AGAINST STATE-OF-THE-ART
Previous related work shows that discrete diffusion models have reasoning potential but lack the training efficiency to rival traditional LLMs (Gulrajani & Hashimoto, 2024). By combining encoder- decoder transformer with diffusion, we can leverage the best of both worlds: training efficiency and
tionally, if the decoder takes in input tokens alongside output from diffusion, it can selectively utilize signals from the diffused latent representation for complex reasoning tasks while discarding noise if the output is not useful (such as when optimizing for Perplexity instead of BertScore). This work does not overlap most prior work on reasoning, making it a suitable add-on to the state-of-the-art reasoning techniques such as Monte Carlo tree search (Xie et al., 2024) or graph of thought (Besta et al., 2024).
239
240
241
242
243
244
245
246 4 RESULTS 247
248 Throughout our experiments, we study the potential benefits and the scalability of our approach
to augment encoder-decoder LLMs with diffusion to enhance reasoning. Specifically, we test this approach against tasks that aim to measure arithmetic reasoning and spatial reasoning. Additionally, we ablate different architecture variants, diffusion sequence length, and model layers to determine the best architecture for scaling. We then summarize our findings to provide insights on how to scale this hybrid architecture.
249
250
251
252
253
254
255
256
257
258
259
260
261 BART for arithmetic tasks. 262
4.1 ARITHMETIC REASONING
To analyze the performance of the proposed hybrid architecture with downstream math tasks, we create single digit addition problem sets where 3-5 single digit numbers are added together. CoT reasoning chains are provided as the target where the model is trained to iteratively add the first two digits. The model is required to output the first token as an answer along with its subsequent reasoning. Table 1 presents comparison between the performance of latent diffusion and fine-tuned
  5

300
301
302
303
304
305
306
307
308
309
310
311
312
313
314
315
316
317
318
319
320
321
322
323
4.2 MOCK SPATIAL REASONING
To study the benefits of latent diffusion augmented LLMs against conventional LLMs, we create a mock spatial reasoning problem where four numbers are presented as input and the model is tasked with coming up with the answer as the first token and subsequent reasoning. The reasoning consists of rotations of up → down → left → right → up. Initially, we start with up, then rotate n times where n is the first number, and each subsequent number reverses the direction of the rotation. For example, given input 1 3, the output should be left. Specifically, the output sequence is left up down up right where left is the final answer. We first start with up, then rotate one time to down, then reverse direction and rotate three times to left.
The problem consists of easy reasoning chains which is required for computing the first token. However, coming up with the first token directly is nearly impossible. In practice, reasoning might not be representable by tokens but the model could still rely on high-level semantics learned by the BART model’s encoder-decoder.
Table 4: Mock spatial reasoning for the rotation task. Architecture Accuracy↑
Latent Diffusion (T=500) 90.4 Latent Diffusion (T=1000) 92.3 Latent Diffusion (T=4000) 89.5
BART (fine-tuned) 0.0
The Encoder-Decoder performs as good as the random baseline throughout training, and predicted the end of text token as the answer near the end of training. We show from the results that latent
Under review as a conference paper at ICLR 2025
 270
271
272
273
274 model capacity and scaling law. 275
We further study the proposed hybrid model’s performance by testing out different arithmetic tasks that mirrors the arithmetic experiments done for GPT-3 (Brown, 2020). We observe that latent diffusion performs remarkably well for its given model size. We acknowledge that this might not be a fair comparison because GPT-3 is not fine-tuned for arithmetic tasks but it should still reflect
276
277
278
279
280
281
282
283
284
285
286
287
288
289
290
291
292
293
294
295
296
297
298 behaved (ordered from easy tokens to hard tokens). 299
Table 2: Double digit additions. Architecture Accuracy↑
  Latent Diffusion (T=1000, 140M) 87.2 BART (fine-tuned) 0.0 GPT-3 (400M) 5.0
GPT-3 (13B) 57.0 GPT-3 (175B) 99.0
Table 3: Single digit three operations.
Accuracy↑ 100.0
GPT-3 (13B) 10.5 GPT-3 (175B) 21.0
Given the same number of training iterations, our findings show that the proposed architecture learns various arithmetic tasks while BART fails completely. The reason is that predicting the first token as answer leads to worse performance for the encoder-decoder because it is unable to self-correct after giving an incorrect answer and have to give subsequent reasoning for the wrong answer (OOD). This is an advantage of diffusion because pretraining data scraped from the internet are rarely well
 Architecture
Latent Diffusion (T=1000, 140M)
 BART (fine-tuned) 11.8 GPT-3 (400M) 2.5
  6

Under review as a conference paper at ICLR 2025
 324
325
326
327
328 in the reasoning chain. For example, if reasoning requires many multiplication arithmetics, diffusion
329 is able to reuse its layers to compute many repetitive multiplications throughout many timesteps,
diffusion does not have to rely on the order of the dataset and can do easy reasoning chains to extrapolate harder answers. Many hard reasoning problems in the real world are impractical to be represented by CoT tokens, therefore, doing reasoning in latent space could be a promising alternative. Augmenting latent diffusion also has an additional benefit when there is many repetition
whereas autoregressive models can only use the same layer once to produce a token.
330
331
332
333
334
335
336
337
338
339 We use the Common Crawl (C4) dataset for all of the ablation experiments since it includes a variety
340 of different sequences from most domains.
4.3 ABLATIONS
We first ablate different architectures to incorporate input text conditioning since text latents could have different properties compared to image class labels (Table 5 presents the results). Throughout experiments, we found that in-context conditioning has minimal compute overhead, while having negligible difference on BertScore, hence we adopt in-context conditioning for most of the ablations.
341
342
343
344
345 Cross-Attention · An additional cross-attention module is added after self-attention for each
346 DiT block to incorporate input text conditioning.
377
Since experiments are trained with different hyperparameters for different learned generation lengths, metrics cannot be compared between different experiments. Further research on how to
In-Context · We concatenate the noised target representation sequence with the input representation sequence. The output is split into two sequences, where the first one is the model output, and the second one is the predicted variance.
347
348
349
350
351
352
353 We further observe improved performance with increasing depth. However, we observe a weak
AdaLN-Zero · Input text conditioning information is incorporated by adding it to the timestep representation which is fed into the AdaLN-Zero block similar to the DiT architecture for class labels.
negative correlation between both metrics and loss with increasing diffusion sequence length which is in contrast to image diffusion transformers. This suggests that further architecture improvements can be made or scaling should be done through increasing layers and not sequence length. We further observe that BertScore does not always correlate with Perplexity, which leads us to hypothesize that the loss function that optimizes representation could sacrifice coherence for semantic similarity. Hence we use BertScore as the main metric for determining performance since it also correlates better with loss, whereas Perplexity has very high variance and depends significantly on the target sequence length and architecture. Images are known to be more parallelizable since there are more independent patches whereas text data are more interdependent.
354
355
356
357
358
359
360
361
362
363
364
365
366
367
368
369
370
371
372
373
374 on average and that shorter sequences have lower Perplexity on average. One hypothesis is that the
375 diffusion model only denoises signals from earlier tokens since they have lower variance (e.g. the
376 next token is easier to predict than the 16th token), leading to later positions denoised as paddings.
To further study the effects of high-level semantics learned by the encoder-decoder architecture, we compare the performance of BART-base (140M parameters) and BART-large (406M parameters) to determine whether the improved quality of both low- and high-level representations also carries over after augmenting with LDMs of the same size. The results show that diffusing better representations from pretrained weights improves BertScore.
We highlight that for this experiment, instead of compressing the last representation of the encoder, we compress the concatenation of the first and last representation of the encoder. This is due to the observation that the decoder did not provide an accurate reconstruction of the original text from only the last encoder representation of BART-large.
We’ve found that generation length decreases with longer training times from preliminary experiments which led to uncomparable BertScores since longer sequences have higher BertScores
7

Under review as a conference paper at ICLR 2025
 378
379 Table 5: Model, architecture, sequence length, and model depth ablation studies
Layers Sequence Length BertScore↑ 6 16 70.0
12 16 70.2
24 16 70.6
6 16 70.0
6 32 70.2
6 64 69.6
6 16 70.2
12 16 70.4
24 16 70.4
6 16 70.2
6 32 70.0
6 64 68.1
6 16 69.7
12 16 69.8
24 16 70.1
6 16 69.7
6 32 69.0
6 64 70.1
 380
381
382
383
384
385
386
387
388
389 Cross-Attention
390 Cross-Attention
391 Cross-Attention
Architecture In-Context In-Context In-Context In-Context In-Context In-Context Cross-Attention Cross-Attention Cross-Attention
 AdaLN-Zero AdaLN-Zero AdaLN-Zero AdaLN-Zero AdaLN-Zero AdaLN-Zero
392
393
394
395
396
397
398
399
400 Table 6: Comparison between BART-base and BART-large. 401
 Model BART-base BART-large
BertScore↑ 67.64 69.80
402
403
404
405
406
407
408
409
410
411
412
413
414
415
416
417
418
419 et al., 2021). By jointly training the unconditional pθ (x) and conditional pθ (x|c) model for a specific
420 class c, we can sample using a linear combination of the score estimates. This is relatively straight-
421 forward to implement by randomly setting pθ(x|c = ∅) during training. Classifier-free guidance
5.2 MIN-SNR-γ
Min-SNR-γ (Hang et al., 2023) improves the training efficiency by weighing each loss term as
 better evaluate these variable length diffusion models will be required to improve the reliability of current metrics.
5 IMPLEMENTATION DETAILS
Throughout our experiments, we adopt classifer-free guidance to improve sample quality at the
expense of sample diversity. We also use Min-SNR-γ because it improves the training efficiency. 5.1 CLASSIFIER-FREE GUIDANCE
Classifier-free guidance is widely known to improve sample quality. (Ho & Salimans, 2022; Nichol
can be used to encourage the sampling procedure such that log p(c|x) is high and tradeoff between sample quality and diversity.
422
423
424
425
426
427
428 wt = min{SNR(t), γ} where t is the timestep and γ is a hyperparameter. By taking into account
429 the signal-to-noise ratio (SNR), min-SNR-γ is better able to traverse the loss landscape by weighing
430 conflicting gradients between earlier and later diffusion steps. Furthermore, Min-SNR-γ takes the
431
minimum between SNR(t) and γ to avoid the model focusing too much on small noise levels. All training run uses γ = 5 as our weighing strategy.
8

Under review as a conference paper at ICLR 2025
 432
433
434
435
436
437
438 binatorial explosion as more tokens are diffused at once, hence the gradients are not as well-behaved
439 (noisy gradient landscapes). As research on diffusion continues, we should expect that it will
6 DISCUSSION
It has been known that diffusion language models yield better diversity when generating text. We show from our work that augmenting latent diffusion with language models outperforms autore- gressive models for certain reasoning cases. A notable limitation of diffusion is that it is relatively inefficient to train compared to conventional language models. One hypothesis is that there is a com-
play a more prominent role in natural language processing to address some of our current limitations.
440
441
442
443
444 feasible approach towards artificial general intelligence (AGI) because it has more diverse ideas,
445 and it can directly implement the programs by reasoning about the structure of the code in latent
An exciting research direction would be to utilize the proposed architecture for idea generation while implementing the ideas with the same architecture specialized for coding. This could be a
space beforehand. This could initiate recursive self-improvement, leading to increasingly automated deep learning research. However, due to its inefficiencies and other potential obstacles, it remains uncertain how far we can practically scale such architectures with current hardware and algorithms.
446
447
448
449
450
451
452
453 demonstrated that augmenting latent diffusion with encoder-decoder architecture outperforms au-
454 toregressive language models in scenarios where tokens have different levels of difficulty (more
7 CONCLUSION
Reasoning involves extrapolating across many facts over extended horizons. In this paper, we
reasoning required), and that adhering strictly to the sequential order of the dataset is not beneficial for accuracy. We propose that this architecture offers a promising approach for solving real-world reasoning tasks by operating in latent space. To our knowledge, this is the first work exploring the augmentation of latent diffusion for reasoning. As research on diffusion models continue to narrow the gap with autoregressive models, we are optimistic that this new architecture can achieve better reasoning with further scale and advancements.
455
456
457
458
459
460
461
462 REFERENCES 463
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale- man, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
464
465
466
467
468
469
470
471
472
473
474
475
476
477
478 Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdi-
Gregor Bachmann and Vaishnavh Nagarajan. The pitfalls of next-token prediction. arXiv preprint arXiv:2403.06963, 2024.
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gian- inazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 17682–17690, 2024.
Tom B Brown. Language models are few-shot learners. arXiv preprint arXiv:2005.14165, 2020. Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer. Advances in
Neural Information Processing Systems, 35:11079–11091, 2022.
nov. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint
479
480
481
482
483
484 Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining
485
vances in Neural Information Processing Systems, 36, 2024.
Guo. Efficient diffusion training via min-snr weighting strategy. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, pp. 7441–7451, 2023. 9
arXiv:1901.02860, 2019.
Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. Ad-

539
Pathak, Laurent Sifre, Morgane Rivie`re, Mihir Sanjay Kale, Juliette Love, et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.
Under review as a conference paper at ICLR 2025
 486
487
488
489
490
491
492
493
494
495 Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey.
Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. Diffusion- bert: Improving generative masked language models with diffusion models. arXiv preprint arXiv:2211.15029, 2022.
Jonathan Ho and Tim Salimans. arXiv:2207.12598, 2022.
Classifier-free diffusion guidance.
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in
neural information processing systems, 33:6840–6851, 2020. arXiv preprint arXiv:2212.10403, 2022.
496
497
498
499
500
501 Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-
DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block- recurrent transformers. Advances in neural information processing systems, 35:33248–33261, 2022.
wise preference optimization for long-chain reasoning of llms, 2024. URL https://arxiv. org/abs/2406.18629.
502
503
504
505
506
507
508
509
510
511
512
513
514
515
516
517
518
519
520
521
522
523
524
525 Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea
526 Finn. Direct preference optimization: Your language model is secretly a reward model. Advances
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pre- training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.
Justin Lovelace, Varsha Kishore, Yiwei Chen, and Kilian Q Weinberger. Diffusion guided language modeling. arXiv preprint arXiv:2408.04220, 2024a.
Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent dif- fusion for language generation. Advances in Neural Information Processing Systems, 36, 2024b.
Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.
Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pp. 8162–8171. PMLR, 2021.
William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
in Neural Information Processing Systems, 36, 2024.
527
528
529
530
531
532 Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv
Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J Nay, Kshitij Gupta, and Aran Komatsuzaki. Arb: Advanced reasoning benchmark for large language models. arXiv preprint arXiv:2307.13692, 2023.
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjo ̈rn Ommer. High- resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF confer- ence on computer vision and pattern recognition, pp. 10684–10695, 2022.
preprint arXiv:2202.00512, 2022.
533
534
535
536
537
538 Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya
10
arXiv preprint

Under review as a conference paper at ICLR 2025
 540
541
542
543
544 Paul Vicol, Luke Metz, and Jascha Sohl-Dickstein. Unbiased gradient estimation in unrolled com-
545 putation graphs with persistent evolution strategies. In International Conference on Machine
Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu. Toward self- improvement of llms via imagination, searching, and criticizing, 2024. URL https://arxiv. org/abs/2404.12253.
Learning, pp. 10553–10563. PMLR, 2021.
546
547
548
549
550
551
552
553
554
555 Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang,
556
557
558
559
560
561
562
563
564
565
566
567
568
569
570
571
572
573
574
575
576
577
578
579
580
581
582
583
584
585
586
587
588
589
590
591
592
593
Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023.
Jiacheng Ye, Shansan Gong, Liheng Chen, Lin Zheng, Jiahui Gao, Han Shi, Chuan Wu, Zhenguo Li, Wei Bi, and Lingpeng Kong. Diffusion of thoughts: Chain-of-thought reasoning in diffusion language models. arXiv preprint arXiv:2402.07754, 2024.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P Lillicrap, Kenji Kawaguchi, and Michael Shieh. Monte carlo tree search boosts reasoning via iterative preference learning. arXiv preprint arXiv:2405.00451, 2024.
A A.1
TRAINING HYPERPARAMETERS ABLATING ARCHITECTURE
 Hyperparameters Diffusion sequence length Depth
Batch size
Sequence Length
Latents sequence length Latents dim
Hidden size
Number of heads
Total timesteps (T) Learning rate
Iterations
Floating Point
ABLATING DEPTH
Hyperparameters
Layers
Architecture
Diffusion sequence length Batch size
In-Context Cross-Attention 16 (each input) 32
12 128 64
8 256 768 12 1000 1e-4 200k float32
12 Layers 12
In Context 16 128 128
16 256 768
12 1000 1e-4
AdaLN-Zero 32
 A.2
 Sequence Length
Latents sequence length
Latents dim
Hidden size
Number of heads
Total timesteps (T)
Learning rate
Iterations 200k
Floating Point
float16
6 Layers 6
24 Layers 24
 11

Under review as a conference paper at ICLR 2025
 594
595
596
597
598
599
600
601
602
603
604
605
606
607
608
609
610
611
612
613
614
615
616
617
618
619
620
621
622
623
624
625
626
627
628
629
630
631
632
633
634
635
636 Hyperparameters
A.3
ABLATING DIFFUSION SEQUENCE LENGTH
 Hyperparameters Diffusion sequence length Batch size
Sequence Length
Latents sequence length Latents dim
Hidden size
Number of heads
Layers
Total timesteps (T) Learning rate
Iterations
Floating Point
In Context 16 16
In Context 32 32
128
128
16
256
768
12
6
1000 1e-4 200k float16
In Context 64 64
 A.4
ABLATING ENCODER REPRESENTATIONS
Hyperparameters Diffusion sequence length Batch size
Sequence Length
Latents sequence length Latents dim
Hidden size
Number of heads
Layers
Total timesteps (T) Learning rate
Iterations
Floating Point
 BART-base T5-large 64
128
128
16
256
768
12
18
1000
1e-4
200k
bfloat16
 A.5
MOCK SPATIAL REASONING FOR THE ROTATION TASK.
 LD (T=500)
LD (T=1000) 16
LD (T=4000)
Encoder-Decoder -
-
-
-
-
-
-
- BART-base 128
1e-4
 Diffusion sequence length Sequence Length
Latents sequence length Latents dim
637
638
639
640
641
642
643
644
645
646 Learning rate
647 Iterations 500k 500k 500k 500k
Hidden size
Number of heads
Layers 24
Total timesteps (T) Pretrained Encoder-Decoder Batch size
500 1000 BART-base BART-base 128 128 1e-4 1e-4
4000 BART-base 128 1e-4
Floating Point bfloat16
bfloat16 bfloat16 bfloat16
12
128
 16
256
768
12

Under review as a conference paper at ICLR 2025
 648
649
650
651
652
653
654
655
656
657
658
659
660
661
662
663
664 A.7 665
666
667
668
669
670
671 Hidden size
672 Number of heads
673 Layers 24
A.6
BIG TRAINING RUN
Hyperparameters Main Run Architecture Cross-Attention Diffusion sequence length 32 Sequence Length 128 Latents sequence length 16 Latents dim 256 Hidden size 768 Number of heads 12 Layers 24
Total timesteps (T) 1000
  Pretrained Encoder-Decoder Batch size
Learning rate
Floating Point
MULTISTEP ADDITION
BART-base 128 1e-4 bfloat16
LD (T=4000)
 674
675
676
677
678
679
680
681
682
683
684
685
686
687
688
689
690
691
692
693
694
695
696
697
698
699
700
701
1000 BART-base 128 1e-4 500k bfloat16