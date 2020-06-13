# Tensor
https://pytorch.org/docs/stable/tensors.html
- `torch.squeeze(input, dim=None, out=None) → Tensor` :Returns a tensor with all the dimensions of input of size 1 removed.

  For example, if input is of shape: (A \times 1 \times B \times C \times 1 \times D)(A×1×B×C×1×D) then the out tensor will be of shape: (A \times B \times C \times D)(A×B×C×D) .

  When dim is given, a squeeze operation is done only in the given dimension. If input is of shape: (A \times 1 \times B)(A×1×B) , squeeze(input, 0) leaves the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the shape (A \times B)(A×B) .

- torch views: PyTorch allows a tensor to be a View of an existing tensor. View tensor shares the same underlying data with its base tensor. Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.
   ```
   t = torch.rand(2, 3)
   tensor([[ 0.3466,  0.2140,  0.9704],
        [ 0.4691,  0.6274,  0.4447]])
   
   t.view(-1,6)
   
   tensor([[ 0.3466,  0.2140,  0.9704,  0.4691,  0.6274,  0.4447]])
   
   t.view(6,-1)
   
   tensor([[ 0.3466],
        [ 0.2140],
        [ 0.9704],
        [ 0.4691],
        [ 0.6274],
        [ 0.4447]])
   
   ```
- `tensor.detach()`: Returns a new Tensor, detached from the current graph. The result will never require gradient.
- `tensor.max(input, dim)`: returns two tensors, max values & idex
   ```
   x = torch.tensor([[0,1,2],[0,11,22]])
   torch.max(x, 0)[0] => tensor([  0,  11,  22])
   x.max(1)[0] => tensor([  2,  22])

   ```
- `torch.unsqueeze(input, dim) → Tensor`: Returns a new tensor with a dimension of size one inserted at the specified position.
    ```
        x = torch.tensor([1, 2, 3, 4])
        torch.unsqueeze(x, 0) => tensor([[ 1,  2,  3,  4]])

        torch.unsqueeze(x, 1) => tensor([[ 1],
                                        [ 2],
                                        [ 3],
                                        [ 4]])
    ```
- `torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor ` : Gathers values along an axis specified by dim.
 For a 3-D tensor the output is specified by:
    ```
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    ```
    example:
    ```
    >>> t = torch.tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
    ```

- `torch.cat(tensors, dim=0, out=None) → Tensor`: Concatenates the given sequence of seq tensors in the given dimension.
    ```
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
            -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
            -0.5790,  0.1497]])
    ```
- `torch.where(condition, x, y) → Tensor`: Return a tensor of elements selected from either x or y, depending on condition.
    ```
    >>> x = torch.randn(3, 2)
    >>> y = torch.ones(3, 2)
    >>> x
    tensor([[-0.4620,  0.3139],
            [ 0.3898, -0.7197],
            [ 0.0478, -0.1657]])
    >>> torch.where(x > 0, x, y)
    tensor([[ 1.0000,  0.3139],
            [ 0.3898,  1.0000],
            [ 0.0478,  1.0000]])
    ```
# Optim
- optim.zero_grad(): Clears the gradients of all optimized torch.Tensor s.
    ```
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    ```
# Loss
https://pytorch.org/docs/stable/nn.html#torch.nn.MSELoss
- torch.nn.MSELoss
- loss = F.mse_loss(Q_expected, Q_targets)
- 'BCEWithLogitsLoss': This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
