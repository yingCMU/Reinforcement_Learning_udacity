# Tensor
https://pytorch.org/docs/stable/tensors.html

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
