std::vector<torch::Tensor> outputlayer_cuda_backward(torch::Tensor input, torch::Tensor grad_indexs, torch::Tensor grad_values, torch::Tensor batch_indexs, torch::Tensor weight){
    const int input_dimension = input.size(1);
    const int batch_size = input.size(0);
    const int grad_size = grad_indexs.size(1);
    const int threads = 1024;
    const dim3 blocks((input_dimension + threads -1)/threads, batch_size);
    auto grad_input = torch::zeros_like(input);
    AT_DISPATCH_FLOATING_TYPES(grad_values.type(), "outputlayer input backward", ([&]{
        outputlayer_cuda_input_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_indexs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            batch_indexs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            input_dimension,
            batch_size,
            grad_size
        );
    }));
    auto grad_weight = torch::zeros_like(weight);
    const int blocks1 = (grad_size + threads - 1)/threads;
    AT_DISPATCH_FLOATING_TYPES(grad_values.type(), "outputlayer weight backward", ([&]{
        outputlayer_cuda_weight_backward_kernel<scalar_t><<<blocks1, threads>>>(
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_indexs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grad_weight.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            input_dimension,
            grad_size
        );
    }));
    auto grad_bias = torch::zeros({weight.size(1)}, at::device(at::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(grad_values.type(), "outputlayer bias backward", ([&]{
        outputlayer_cuda_bias_backward_kernel<scalar_t><<<blocks1, threads>>>(
            grad_indexs.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            grad_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grad_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grad_size
        );
    }));
    return {grad_input, grad_weight, grad_bias};
}

__global__ void outputlayer_cuda_input_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> grad_indexs,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_values,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> batch_indexs,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> weight,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_input,
    const int32_t input_dimension,
    const int32_t batch_size,
    const int32_t grad_size){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < batch_size && col < input_dimension){
        float sum = 0.0;
        for(int k = batch_indexs[row][0]; k <= batch_indexs[row][1]; k++){
            sum += grad_values[k] * weight[col][grad_indexs[1][k]];
        }
        grad_input[row][col] = sum;
    }

}

__global__ void outputlayer_cuda_weight_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> grad_indexs,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_values,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_weight,
    const int32_t input_dimension,
    const int32_t grad_size){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < grad_size){
        for(int k = 0; k < input_dimension; k++){
            grad_weight[k][grad_indexs[1][col]] += grad_values[col] * input[grad_indexs[0][col]][k];
        }
    }
}
__global__ void outputlayer_cuda_bias_backward_kernel(
    const torch::PackedTensorAccessor32<int32_t,2,torch::RestrictPtrTraits> grad_indexs,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_values,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> grad_bias,
    const int32_t grad_size){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < grad_size){
        grad_bias[grad_indexs[1][col]] += grad_values[col];
    }
}