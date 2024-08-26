# Intro

LayerNorm(LN) is used instead of BatchNorm, no one knows why it works but improves numerical stability but can't be turned off in inference.

# Folding

Folding LN helps make it lower overhead to deal with. The following flags default to true in the hookedtransformer.from_pretrained:

- center_writing_weights
- fold_in

Intuitively, LayerNorm acts on each residual stream vector (ie for each batch element and token position) independently, sets their mean to 0 (centering) and standard deviation to 1 (normalizing) (_across_ the residual stream dimension - very weird!), and then applies a learned elementwise scaling and translation to each vector.

Mathematically, centering is a linear map, normalizing is _not_ a linear map, and scaling and translation are linear maps.

- **Centering:** LayerNorm is applied every time a layer reads from the residual stream, so the mean of any residual stream vector can never matter - `center_writing_weights` set every weight matrix writing to the residual to have zero mean.
- **Normalizing:** Normalizing is not a linear map, and cannot be factored out. The `hook_scale` hook point lets you access and control for this.
- **Scaling and Translation:** Scaling and translation are linear maps, and are always followed by another linear map. The composition of two linear maps is another linear map, so we can _fold_ the scaling and translation weights into the weights of the subsequent layer, and simplify things without changing the underlying computation.

A fun consequence of LayerNorm folding is that it creates a bias across the unembed, a `d_vocab` length vector that is added to the output logits - GPT-2 is not trained with this, but it _is_ trained with a final LayerNorm that contains a bias.
