import torch

class Conv2D:
    def __init__(self, kernel):
        self.kernel = kernel  # Initialize with a specific kernel

    def apply_kernel(self, image):
        # Get the dimensions of the image and the kernel
        ri, ci = image.shape  # Image dimensions
        rk, ck = self.kernel.shape  # Kernel dimensions

        # Compute output dimensions
        ro, co = ri - rk + 1, ci - ck + 1  # Output dimensions

        # Initialize an empty tensor for the output
        output = torch.zeros([ro, co])

        # Apply the kernel to the image (convolution operation)
        for i in range(ro):
            for j in range(co):
                output[i, j] = torch.sum(image[i:i+rk, j:j+ck] * self.kernel)

        return output