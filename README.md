**Project Overview**

This repository implements a small Transformer-based sentiment classifier (training pipeline and notebook examples). The core experiment lives in [EncoderBlock.ipynb](EncoderBlock.ipynb) which shows data preparation, model definition, training loop, and evaluation.

**Note**: While interacting with [EncoderBlock.ipynb](EncoderBlock.ipynb) , you dont need to install the dataset it will load the dataset in main memory after running the third cell 

**Quick Start**
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Run notebook:** open [EncoderBlock.ipynb](EncoderBlock.ipynb) and run cells in order to reproduce training and evaluation.  
- **Main model files:** see [models/model/encoder.py](models/model/encoder.py) and [models/embeddings/transformer_econding.py](models/embeddings/transformer_econding.py).

**Model Flow**
- **Input:** tokenized `input_ids` (shape (B, T), dtype `torch.long`).  
- **Embedding:** token + positional embeddings → (B, T, d_model) (float).  
- **Encoder stack:** a sequence of `EncoderLayer` modules that process float embeddings and preserve shape (B, T, d_model).  
- **Pooling & Classifier:** after the final encoder layer, pool (mean over non-pad tokens or use a CLS token) and run a single linear classifier to obtain logits (B, num_classes).

**Layers — What They Do**
- **Token & Positional Embeddings:** map integer token indices to float vectors. See [models/embeddings/token_embeddings.py](models/embeddings/token_embeddings.py) and [models/embeddings/positional_encoding.py](models/embeddings/positional_encoding.py).
- **EncoderLayer:** attention + feed-forward block that inputs and outputs float tensors (see [models/blocks/encoder_layer.py](models/blocks/encoder_layer.py)).
- **ClassifierHead:** a simple `nn.Linear` from `d_model` → `num_classes` (see `ClassifierHead` in [EncoderBlock.ipynb](EncoderBlock.ipynb)).

Important constraint: embeddings must receive integer index tensors (`torch.long`). Do not call `nn.Embedding` on float tensors — that causes the RuntimeError: "Expected tensor for argument #1 'indices' ... but got torch.cuda.FloatTensor instead."

**Why you might see shape/dtype errors**
- Passing encoder outputs (float, shape (B,T,d_model)) into `CrossEntropyLoss` directly will fail because the loss expects logits shape (B, num_classes). Make sure to call the classifier on pooled encoder outputs before computing loss.  
- If you see an embedding-type error, check `input_ids.dtype` and where `nn.Embedding` is called.

**How to Collect Layer Outputs (optional)**
- To inspect intermediate representations, collect each layer output into `hidden_states` and optionally `torch.stack(hidden_states)` to get shape (n_layers, B, T, d_model). This is useful for analysis or auxiliary losses but increases memory usage.
- Memory-saving option: pool each layer (B, d_model) and store (B, n_layers, d_model) instead of full (B, T, d_model).

**Fixing Overfitting — Contribution Guide**
- **Diagnostics:** first check training vs validation loss/accuracy curves (notebook already collects `train_losses`, `val_losses`). If training accuracy is much higher than validation accuracy, overfitting is likely.
- **Quick fixes to try:**  
	- Increase dropout (`drop_prob`) in embedding/encoder layers.  
	- Use weight decay (the code already supports `weight_decay` in AdamW).  
	- Reduce model capacity: decrease `d_model`, `n_layers`, or `d_ff`.  
	- Add early stopping or limit epochs.  
	- Use data augmentation or more training data.  
	- Use gradient clipping and lower learning rate or a different LR schedule.
- **Where to change:** most hyperparameters live in `conf.py` and the model initializers in [EncoderBlock.ipynb](EncoderBlock.ipynb) and [models/model/encoder.py](models/model/encoder.py).
- **To contribute a fix:** open a PR that includes (a) a short description, (b) a notebook cell or test that reproduces the issue and shows improvement, and (c) code changes. Add tests or a short training script demonstrating reduced val loss.

**Directories — What `EncoderBlock.ipynb` Uses**
- Used by the main notebook and model: [models/](models/) and its subfolders: [models/embeddings](models/embeddings), [models/blocks](models/blocks), [models/model](models/model).  
- Utility helpers used: [util/](util/) for tokenizers and data loading.  
- Checkpoints are saved under: `saved_model/` (used to save/load weights).

**Directories Structure**
-[`encoder_layer.py`](models/blocks/encoder_layer.py)] : for the Full Encoder Block  
-[`multihead_attention.py`](models/layers/multihead_attention.py)] : for the implementation of the MultiHead Attention Layer
-[`add_norm.py`](models/layers/add_norm.py)] : for the implementation of the Add&Norm Layer
-[`positionwise_feed_forward.py`](models/layers/positionwise_feed_forward.py)] : for the Implementation of the FFN layer 

**Directories You Can Reuse / Extend**
- Some folders are present but not required directly by `EncoderBlock.ipynb`. These are useful places to add features or experiments:  
	- `Refrence/` — example notebooks you can port improvements from.  
	- `images/` — visuals for README / analysis.  
	- Any other project folders not referenced in `EncoderBlock.ipynb` are fine places to add upgrade utilities (data augmentation, monitoring scripts, alternative model variants).

**How to Contribute**
- Fork the repo, make a branch, implement changes, add a short notebook cell showing before/after metrics, and open a PR. In the PR description, explain whether the change targets overfitting (hyperparams, architecture, regularization) or model correctness (dtype/shape fixes).

**Contact / Notes**
- If you run into dtype or shape errors, verify the tensor shapes and dtypes at the embedding and loss boundaries: print `input_ids.shape, input_ids.dtype` before embedding and `encoder_out.shape` before pooling/classifier.

Thanks — contributions welcome!  

