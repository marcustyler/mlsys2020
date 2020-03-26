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

## PLink: Datacenter Network Locality for Efficient Parallel Training

- AllReduce algorithms fall short of optimal performance in datacenters
- Use locality: find clusters of VMs via network latency. Embed VMs on 2d plane and create a map
- Use "Autotune" to detect changes in network and adjust nodes/balance the schedule

## Federated Optimization

- Federated training: privacy-preserving training

## Scaling Back-Propagation (BP)

- Scale BP by Parallel Scan Algorithm (BPPSA)
- BP has strong sequential dependency
- Blelloch Scan Algorithm

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

## FLEET: Flexible Ensemble Training

- When sharing data batches, all models in ensemble held back by slowest model
    - Solution: Allocate different numbers of GPUs to different DNNs (more for slower models), called a "flotilla"
    - Creating flotillas is an NP-hard problem. So dynamically determine flotillas with a Greedy Allocation Algorithm, and update flotillas/GPUs with each iteration
    - With multiple GPUs in a model, try to keep them in the same node. When model converges, release these GPUs and continue training the other models

## Checkmate: Breaking Memory Wall with Optimal Tensor Rematerialization

- BigGAN, VideoBERT... models are exploding in size and reaching memory capacity
- There is a training time/memory tradeoff
- In training there is the Forward Pass (convolutions, pooling, etc.) and the Backward Pass (stashed activations). Rather than keeping all nodes/layers in memory, recompute them during the Backward Pass
- Problems: Time/memory vary between layers, and nonlinear DNNs
- Solutions: Have an accurate cost model and flexible search space. Use linear programming to minimize forward and backward cost. Constrain search to valid/acceptable solutions
- Code: checkmateai.github.io

# Keynote: Chris Re (Stanford) (Monday 3/2/2020)

- Can we build a mathematical and systems structure for messy data streams?
- Ex.: Triage
    - Augment chest scans
    - Label quality and quantity > model choice
    - Data augmentation is critical
- Programmatic labels: Noisy but generates training data via weak supervision
- Beware data artifacts such as "drains" in x-rays of chests with pneumonia, which are anti-causal (occur AFTER diagnosis). Deep learning is even better at picking up artifacts
- Code: snorkel.org

# Session 3: Efficient Inference and Model Serving (Monday 3/2/2020

## Neural Network Pruning

- Cycle between Modeling / Pruning/ Finetuning until ready to evaluate
- This increases the efficiency of the model (with a tradeoff of quality)
- Difficult to compare work in this area, with single datapoints per paper and no standard architecture. Also no standard definition of "compression ratio", or of FLOPs
- ShrinkBench

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
    - Q/K are derived from atributes (query/key)
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

## OPTIMOS

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

