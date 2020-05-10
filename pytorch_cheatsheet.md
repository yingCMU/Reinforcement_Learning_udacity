

https://pytorch.org/docs/stable/tensors.html

- tensor.detach(): Returns a new Tensor, detached from the current graph. The result will never require gradient.
- tensor.max(input, dim): returns two tensors, max values & idex
   ```
   x = torch.tensor([[0,1,2],[0,11,22]])
   torch.max(x, 0)[0] => tensor([  0,  11,  22])
   x.max(1)[0] => tensor([  2,  22])

   ```
- torch.unsqueeze(input, dim) → Tensor: Returns a new tensor with a dimension of size one inserted at the specified position.
    ```
        x = torch.tensor([1, 2, 3, 4])
        torch.unsqueeze(x, 0) => tensor([[ 1,  2,  3,  4]])

        torch.unsqueeze(x, 1) => tensor([[ 1],
                                        [ 2],
                                        [ 3],
                                        [ 4]])
    ```
- torch.gather(input, dim, index, out=None, sparse_grad=False) → Tensor  : Gathers values along an axis specified by dim.
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
- optim.zero_grad(): Clears the gradients of all optimized torch.Tensor s.
    ```
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    ```
