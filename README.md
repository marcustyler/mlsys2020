# MLSys 2020

# Session 1: Distributed and Parallel Learning Algorithms (Monday 3/2/2020)

## Massively Parallel Hyperparameter Tuning

- AutoML
- Large models, large search spaces -> long training times
- Large-scale Regime: evaluate # models >> # workers
    - Use "early stopping" to increase # evaluations
    - Identify "losing" hyperparameters and stop them before they finish fitting
    - Approach: Successive Halving. Iteratively remove underperforming models when running all in parallel
    - Problems: Non-monotonic, non-smooth, different convergence rates for different models. Stragglers hold others back.
    - Solution: Asynchronous SHA (ASHA) to deal with stragglers - run a subset and promote when possible.
- Impressive list of APIs to work with existing frameworks (sklearn, TF, PyTorch etc.)
- [Web site](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/) and [paper](https://arxiv.org/abs/1810.05934)

## PLink: Datacenter Network Locality for Efficient Parallel Training

- AllReduce algorithms fall short of optimal performance in datacenters
    - Up to 90% of training time consumed
    - Intra-rack latency < cross-rack latency
    - Approach: aggregate gradients intra-rack and then only push one aggregation cross rack
- Use locality: find clusters of VMs via network latency. Embed VMs on 2d plane and create a map
- Use "Autotune" to detect changes in network and adjust nodes/balance the schedule
- [Paper](https://proceedings.mlsys.org/static/paper_files/mlsys/2020/33-Paper.pdf)

## Federated Optimization

- Federated training: privacy-preserving training
- Provide theoretical convergence guarantees

## Scaling Back-Propagation (BP)

- Scale BP by Parallel Scan Algorithm (BPPSA)
- BP has strong sequential dependency
- Blelloch Scan Algorithm O(2 log N) vs O(N)
- Also exploit jacobian sparsity
- Deep nets see ~2X speedup

## Distributed Hierarchical Parameter Server for Ads Systems

- How to handle TB of features?

# Session 2: Efficient Model Training (Monday 3/2/2020)

## Resource Elasticity in Distributed Deep Learning

- Autoscaling System
    - Scaling heuristics, straggler detection
    - Add and replace workers
    - Parameter synchronization
- Heuristics
    - Utility vs financial cost function provided by user

## SLIDE: Smart Algorithms for Deep Learning

- Uses probabilistic hashing techniques to approach GPU speed on CPUs

## FLEET: Flexible Ensemble Training

- When sharing data batches, all models in ensemble held back by slowest model
    - Solution: Allocate different numbers of GPUs to different DNNs (more for slower models), called a "flotilla"
    - Creating flotillas is an NP-hard problem. So dynamically determine flotillas with a Greedy Allocation Algorithm, and update flotillas/GPUs with each iteration
    - With multiple GPUs in a model, try to keep them in the same node. When model converges, release these GPUs and continue training the other models

## Checkmate: Breaking Memory Wall with Optimal Tensor Rematerialization

- BigGAN, VideoBERT... models are exploding in size and reaching memory capacity
- There is a training time/memory tradeoff
- How can we train larger models than fit in GPU (or increase batch size)?
- In training there is the Forward Pass (convolutions, pooling, etc.) and the Backward Pass (stashed activations). Rather than keeping all nodes/layers in memory, recompute them during the Backward Pass
- Problems: Time/memory vary between layers, and nonlinear DNNs
- Solutions: Have an accurate cost model and flexible search space. Use linear programming to minimize forward and backward cost and find optimal rematerialization schedule. Constrain search to valid/acceptable solutions
- Code: checkmateai.github.io

# Keynote: Theory and Systems for Weak Supervision, Chris Re (Stanford) (Monday 3/2/2020)

- Training data often the bottleneck in DL systems. 
    - SOTA models are free to download and use
    - Hardware is accessible in the cloud to rent
    - How can we commoditize training data? 
- Model differences are overrated and supervision is underrated (data quality more important than model choice)
- Can we build a mathematical and systems structure for messy data streams?
- Ex.: Triage
    - Augment chest scans
    - Label quality and quantity > model choice
    - Data augmentation is critical
- Programmatic labels: Noisy but generates training data via weak supervision
- Let's model the labeling process (snorkel.org)
    - Step 1: Users write noisy labeling functions
    - Step 2: Labeling functions are modeled to discover correlations and denoise
- Key finding: you only need ~2X more auto-generated noisy labels to attain same performance as high-quality labels
- Beware data artifacts such as "drains" in x-rays of chests with pneumonia, which are anti-causal (occur AFTER diagnosis). Deep learning is even better at picking up artifacts
- Code: snorkel.org, dataintegration.ml, "Automating the art of data augmentation" blog post

# Session 3: Efficient Inference and Model Serving (Monday 3/2/2020

## Neural Network Pruning

- Cycle between Modeling / Pruning/ Finetuning until ready to evaluate
- This increases the efficiency of the model (with a tradeoff of quality)
- Difficult to compare work in this area, with single datapoints per paper and no standard architecture. Also no standard definition of "compression ratio", or of FLOPs
- [ShrinkBench](https://shrinkbench.github.io/) tool for automatically evaluating pruning techniques (PyTorch)

## SkyNet: Object Detection on Embedded Systems

- Difficulty with real-time requirement

## Willump: Statistically-Aware Optimizer for ML Inference

- Feature computation: a bottleneck
- Use cheap model on easy cases, and complex model on harder cases ("statistically-aware")
- Can use "Model Cascades" (but traditionally custom and do not generalize)
- Willump generates a cascade, and uses a search algorithm to determine features. First creates an approximate model on most confident datapoints
- E.g. Top-K Approximation: Willump does this automatically

## MNN: Universal & Efficient Inference Engine

- On-device inference
- Kernel optimization: Winograd in MNN

# Session 4: Model/Data Quality and Privacy (Monday 3/2/2020)

## HoloClean: Attention-based Learning for Missing Data Imputation

- Probabilistic inference for data repairing
- Challenges: Missing values are often systematically biased, and may not be missing at random. Mixed types -> mixed distributions
    - Drawbacks: Complex rules, models for imputation etc.
- Missing at Random (MCAR)
- AimNet
    - Exploit structure in data
    - Learn schema-level relationships
    - 1) Encode mixed data values
    - 2) Attention identifies context/schema
    - 3) Predict
- Encoding mixed types: convert to embedded vectors
- Attention layer
    - Q/K are derived from attributes (query/key)
    - E.g. "County" places large weight in zip code, and smaller in city...
    - So we get attention weights
- For prediction, convert vectors back to original value types
- Can predict MCAR/i.i.d. missing data, and robust against systematic bias (the attention schema helps with systematic bias)

## Privacy-Preserving Bandits

- Start with histogram of user interests/page views
- Can you detect if someone was used in training a model?
- Can we quantify privacy? Perhaps from hiding in the crowd?
- Privacy Model: Crowd-blending + i.i.d. Sampling -> Differential Privacy

## Downstream Instability of Word Embeddings

- Prediction churn: small data changes -> large differences in predictions
- Problems with churn: poor user experience, model dependencies, research reliability
- Embedding Server
    - Refreshes periodically
    - Downstream tasks: NER (named entity recognition), Q&A, sentiment analysis, etc.
    - Stability/memory tradeoff
- Downstream disability: % prediction disagreement
- What hyperparameters impact downstream instability?
    - Dimension (# features / word) and precision (# bits / feature)
    - With increasing dimension and precision, disagreement decreases for sentiment analysis and NER on Wikipedia corpus
    - Also, greater overall memory -> lower disagreement
- Introduce Eigenspace Instability Measure (EIS)
    - Similarity of left singular vectors of two embeddings correlates strongly with downstream disagreement
    - Equal expected mean-square difference in disagreement
- Also can use k-NN, semantic displacement, PIP loss, Eigenspace Overlap
- Can use "selection task" to choose parameters for minimal EIS / downstream disagreement
- Also on "ranking" downstream tasks
- mleszczy@stanford.edu (works with Chris Re)

## Model Assertions for Monitoring & Improving MC Models

- Software 1.0 (no ML) has standard quality assurance (QA) methods
- Models can make systematic errors
    - E.g. rapidly changing predictions of a single object between pedestrian/bike makes it much harder to predict the path of the object
    - Assertions via active learning (user labeled) and weak supervision. Also fuzzing (TensorFuzz)
    - How to select data points for active learning? Multiple assertions can flag same datapoints
    - For weak supervision, can use correction rules (e.g. no flickering boxes in video). Then retrain model with new labels
    - Also, different sensors should agree (another assertion rule). Consistency across ID and no abrupt changes

# Session 5: ML Programming Models and Abstractions & ML Applied to Systems (Tuesday 3/3/2020)

## AutoPhase: HLS Phase Orderings with Deep RL

- How to order compiler flags? There are many possibilities with a wide range of potential complexities

## Auto-Batching

- Predictive Pre-compute (Facebook): predict user behavior to save resources and speed up content serving to users (e.g. dog photos)
- Uses RNNs to model hidden states for users

## Predictive Precompute with Recurrent Neural Networks

## Sense & Sensitivities: Algorithmic Differentiation (Language Design for ML)

- Overview of [Zygote](https://github.com/FluxML/Zygote.jl) differentiation tool for Julia language
- Instead of tracing to perform reverse mode autodiff, zygote is "source to source"
- Hooks into Julia compiler to generate source code for backward pass
```julia
julia> using Zygote

julia> f(x) = 5x + 3

julia> f(10), f'(10)
(53, 5)

julia> @code_llvm f'(10)
define i64 @"julia_#625_38792"(i64) {
top:
  ret i64 5
}
```
- Dynamic programming constructs like control flow, recursion, closures, structs, dictionaries are supported
- This is great for scientific programming where we're often trying to satisfy convergence criteria and need to iterate a solution
- If you can write a forward model for a system in Julia, you can differentiate it
- Encode structure in ML model (e.g. conv nets based on biological structures in vision systems)
- Dramatically speed up learning in [control problems](https://fluxml.ai/2019/03/05/dp-vs-rl.html) (vs RL)
    - take advantage of physics models (differentiable equations)
    - treat reward produced by RL env as a differentiable loss (model-free to model-based RL)
- Embed NN models inside differential equations to discover unknown relationships from data
- More info:
    - [Flux](https://fluxml.ai/) Julia's ML toolkit
    - [SciML](https://sciml.ai/) Open source scientific machine learning software (mostly Julia, but some tools have python and R bindings)
    
## Ordering Chaos: Memory-Aware Scheduling

- GPU Sharing Primitives: memory sharing in single GPU lane, for efficient job switching with dynamic scheduling

# Session 6: Efficient Inference and Model Serving (Tuesday 3/3/2020)

## Fine-Grained GPU Sharing Primitives for DL Applications

## Improving Graph NNs with ROC

- Graph sampling loses info
- "Roc" uses no sampling (a distributed multi-GPU method)
    - Attribute parallelism (e.g. breaking an image into quadrants)
    - Dynamic allocation of memory to minimize data transfer between DRAM/GPU
    - Linear regression-based graph partitioner using graph features and hardware features
- github.com/FlexFlow/Roc

## OPTIMUS: OPTImized matrix MUltiplication Structure for Transformer neural network accelerator

- With sparse graph RNNs, load imbalances occur potentially in weight matrices
- Skip redundant decoding computations
- High MAC utilization
- Custom hardware

## PoET-BiN: Power Efficient Tiny Binary Neurons

## Keynote: Cryptography for Safe ML (Shafi Goldwasser) (Tuesday 3/3/2020)

- Ca. 1980s, 1990s: RSA encryption -> there exists c(.) functions that cannot be PAC-learned
- Then came the Learning With Errors problem (LWE) -> homomorphic encryption
- Geometry/lattice-based crypto resilient against quantum methods (as opposed to number theory-based crypto...)
- Now, how to maintain privacy of both people's data and the model? How to be sure model has not been tampered with? (e.g. with malicious training data)
- You have both a Learner (ML model) and a Verifier which communicate back and forth, and can differ in data access
- Challenge: Adversarial ML
    - One solution: allow "no prediction" result to deal with odd adversarial examples, to not misclassify them. But where to draw the line?
    - Or maybe: trace unauthorized use of your data/model and verify if privacy mechanisms were used
- And then: how do we make ML fair and interpretable to all demographics?
- Does secrecy of random seeding matter?
- Garbled circuits, differential privacy, etc. ...
    - Secure multi-party computation: distributed models where parties don't have to share data with each other
    - Fully homomorphic encryption: keep data encrypted
    - e.g. hospital w/ secret model, and client w/ private data; can they together make a prediction/diagnosis?
    - An approach is to use both homomorphic on some CNN layers and multi-party on others (Delphi)
- Current work: how to mitigate error in predictions in encryption

# Session 7: Quantization of Deep Neural Networks (Tuesday 3/3/2020)

## Quantization for Deep NN Inference on Microcontrollers

## Quantization Thresholds for Fixed-Point Inference

## Riptide: Binarized NNs

- Multiplication -> XNOR (w/ +1, -1)
- Hard to simulate runtime on binary NNs
- Riptide: fast modeling, with only two integer operations (add and right-shift). Multiplication is binarized

## Winograd-aware Quantized Networks

# Session 8: Efficient Model Training 2 (Tuesday 3/3/2020)

## Blink: Collectives for Distributed ML

- Faster collective communication protocols

## Systematic Methodology for Analysis of DL

- ParaDnn is their method, similar to MLPerf

## MotherNets: Ensemble Learning

- Combine predictions to reduce variance
- Types
    - Independent training: all data in all (separate) networks
    - TreeNets: start in same nodes then branch out
    - Snapshot Ensembles
- MotherNets - share epochs
    - 1) Construct MotherNet. Capture in each layer nodes with fewest parameters, put nodes into MotherNet
    - 2) Train MotherNet
    - 3) Hatching: function preserving transformation (i.e. transfer trained MotherNet nodes back to ensemble NNs)
    - 4) Finally train ensemble. This is faster than Bagging, Snapshot, TreeNets
    - Can navigate tradeoff between training time and modeling error (by number of clusters; when # is size of ensemble -> independent training)

## MLPerf Training Benchmark

# Workshop on Secure and Resilient Autonomy (SARA) (Wednesday 3/4/2020)

## Keynote: A DARPA View

- Heilmeier Catachism: What are you trying to do? What are the costs? What are the risks?
- 47% of consumers state security/privacy as obstacles to adopting IoT devices. 18% quit using IoT for security reasons
- Sensor challenges: Sensors deployed for long durations, with limited power, and time critical event detections, and different environments (urban, forest, bases...)
- ISEE-3: A satellite that was once redirected to comet orbit (and became ICE-3). It vanished for several decades and they had to rebuild transceiver and signal processing capabilities in order to talk to it when it returned to Earth. The flexible nature of its radio software allowed them to reestablish communications
    - Thus software interfaces w/ the physical world matter
    - But it takes work to develop flexible software (DevSecOps: Development, Operations, Security)
- 5G: Everything is connected, higher bandwidth, millions of devices / km^2
    - Broadband -> cloud computing -> next wave of AI
    - Environmental context -> better decisions
    - Industry has far more money to devote to 5G development than DARPA. But DoD cares about the privacy of people, and resilient systems. DoD uses tech with lifetimes of decades, vs. 2-3 years for commercial
    - Bit-rot: When old software stops working
    - Now DARPA is pushing for open and transparent software, and programmable (and secure) software, which increases flxibility (OPS-5G)
    - How to use good encryption/security w/ limited power? This issue and 5G are the current DARPA focus
    - Open Radio Access Network (O-RAN) Alliance: Building open, standard interfaces in 5G-RAN
    - Fully Homomorphic Encryption (FHE): Information can be processed while encrypted. Tends to be quite slow

## Circuits for Entropy Generation & Secure Encryption

## Feature Map Vulnerability Evaluation in CNNs

- Tesla has redundant chip -> 2x power, processing etc.
- Software-directed hardening approach
- Feature Maps (Fmap) are robust to translational effects of inputs
- Fmap vulnerabilities evaluation
    - Statistical error injection: Flip bits in the NN. But mismatches are rare. So replace binary view with continuous view (cross entropy loss). Their metric: average delta cross entropy loss

## Reliable Intelligence in Unreliable Environment

- Noisy sensors, non-ideal conditions (rain etc.)
- Must make current AI resilient to non-ideality
- Fusion of multiple sensors (e.g. lidar and RGB)

## Towards Information Theoretic Adversarial Examples

- Perturbations on image -> misclassified
- Carlini and Wanger's attack (minimize L2-norm)
    - Also, Projected Gradient (PGD) attack, Deepfool attack
- Proposed method: Mutual Information Neural Estimation (MINE)
    - Use -I(.) and elements of C & W optimization to create natural-looking adversarial images
    - Gaussian random projects on image -> I(x,x+delta)

## Explaining Away Attacks Against Neural Networks

- Gradient-based adversarial attacks
    - Attribution for ML: What pixels motivate the decision? Use gradients to find out
    - Integrated Gradients (Sundararajon 2017). "Explain 4" (as in MNIST) or "explain 9" shows prediction versus baseline (uniform) probability. You see which pixels were most informative for the given decision. The solution: take path integral between the input and baseline. This shows attributions
    - Proposed variant: SHAP integrated gradients
    - They detect adversarial attacks with 98+% accuracy

## Keynote: Towards Robust and Efficient DL Systems

- Adversarial attacks against DNNs, for m classes and perturbation on a datapoint
    - Fast Gradient Sign Method (FGSM): fast but suboptimal
    - C & W optimization
- Consider a universal attack framework using ADMM (alternating direction method of multipliers)
    - Structured attack demonstrates stronger attack sparsity than C & W. Therefore, they can perturb fewer pixels in ADMM attacks
- Finally can train model with adversarial robustness and also use weight pruning (model compression)

## MUTE: Multi-Hot Encoding for Neural Network Design (IBM)

- For noisy, slightly out of distribution examples, misclassification is likely
- Some classes are more semantically similar than others (e.g. "3" and "8" are similar images)
- MUTE pushes similar classes apart (as seen in t-SNE)
    - Specifically, use sigmoid output with multiple hot bits, instead of softmax one-hot encodings. This forces the model to learn discriminating features during backpropagation, like in error correcting codes
    - Create an inter-class similarity matrix by training autoencoders on every class, and then running other classes through the autoencoders for all pairs
    - Assign a larger Hamming distance to more similar classes. Number of output nodes remains the same, just a more efficient use of binary vector than with one-hot
    - This method worked especially well with "negative" image examples
    - arxiv.org/pdf/1910.07042.pdf

## WARDEN: Deception in Data Centers

- Power systems have large attack spaces, from voltage/freq controls and such
- Power Contention: when a data center draws more power than allowed (from running a power-hungry program on power capped systems)
- Subset of servers can be represented as multidimensional features (e.g. power consumption and storage); can we detect malicious activities?
- Low-complexity codes for confidential data can reduce attack surface

## Self-Progressing Robust Training (Chen, IBM)

- Types of adversarial ML
    - The adversarial T-shirt (wild design/colors, unable to be detected as a person)
    - Also poisoning attack, evasion attack, model injection attacks
- The authors plotted tradeoff between accuracy and l_inf CLEVER score (robustness of model), which are negatively correlated
- Data augmentation with adversarial examples helps but does not fix the problem. Input transformation can be bypassed
- Dirichlet Label Smoothing: draw labels from a distribution y_est = (1 - a)*y + a*Dirichlet(beta)
    - Also include Gaussian Augmentation and Mixup
    - SPROUT algorithm updates smoothing/augmentation params with each epoch
- Customized Adversarial Training (CAT)
    - Not all samples need be treated equally, nor all labels
    - Improves robustness without losing accuracy
    - Model prediction should be less confident for perturbed samples that are farther from x_i


# MLOps workshop

## Holoclean

See above

## Overton, Chris Re (Apple/Stanford)

- Internal AutoML tool at Apple for building and monitoring ML products and deployments
- New models ruled in 2017-18, but marginal improvements from new models since then
- Chris sees the most potential gains from automating "low-margin" jobs that he believes ML engineers shouldn't have to worry about:
    - setting hyperparameters
    - selecting model architecture
    - reducing model size or compute for deployment
- Overton uses a declarative approach to specifying goals (tell me what you want, not how to do it). Inspired by SQL etc
    - Input schema:
        - data payloads (data and types etc)
        - model tasks (classification, regression, etc)
        - input, output, and coarse grained data flow
    - Overton then compiles schema into TF, CoreML, and/or PyTorch graphs
        - performs model search
        - hyperparameter search
        - produces tagged binaries for deployment along with metadata (validation loss etc.) so that deployment can be automated
- In production has reduced errors by 1.7 to 2.9 times

## Auto Compilation of MLOps, Tianqi Chen (UWash, MXNet, XGBoost)

- Apache TVM: end-to-end compiler for ML workloads
- User specifies general computation at a high level, and target hardware
- TVM uses AST and cost model to select best algorithm and implementation 
- Works with newer models (transformers) and IOT edge compute hardware
- Check out NeurIPS '18 paper: "Learning to optimize tensor programs"
