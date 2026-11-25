import math

import pytest
import torch

from src.models.utils import attention, patchify, unpatchify_grid


# create fixtures for testing


def test_attention():
    
    # zero queries
    keys = torch.tensor([[[1.0], [2.0]]])
    queries = torch.zeros(1, 2, 1)
    values = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    expected = torch.tensor([[[4.0, 6.0], [4.0, 6.0]]])
    torch.testing.assert_close(attention(keys, queries, values), expected)
    
    # non trivial example 
    #! AI helped me come up with this example
    log3 = math.log(3.0)
    keys = torch.tensor([[
        [0.0, 0.0, 0.0, 0.0],
        [2 * log3, 0.0, 0.0, 0.0],
    ]])
    queries = torch.tensor([[
        [ 1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
    ]])
    values = torch.tensor([[[2.0, 10.0], [6.0, 2.0]]])

    expected = torch.tensor([[[5.0, 4.0], [3.0, 8.0]]])
    torch.testing.assert_close(attention(keys, queries, values), expected)


def test_patchify_creates_grid_of_patches():
    image = torch.Tensor(2, 4, 4, 2) # batch of 2 images, 4x4 size, 2 channels
    # initialize images with ordered values for testing
    #[[(0,1), (2,3), (4,5), (6,7)],
    # [(8,9), (10,11), (12,13), (14,15)],
    # [(16,17), (18,19), (20,21), (22,23)],
    # [(24,25), (26,27), (28,29), (30,31)]]
    image[0] = torch.arange(4*4*2).reshape(4, 4, 2)
    image[1] = torch.arange(4*4*2).reshape(4, 4, 2) + 100
    
    
    grid_shape = (2, 2) # grid of 2x2 patches
    patches = patchify(image, grid_shape, overlap=0) 
    assert patches.shape == (2, 2, 2, 2, 2, 2) # (batch_size, n_patches_x, n_patches_y, patch_height, patch_width, channels)
    
    # fabricate expected patches for the first image
    expected_patches_image0 = torch.tensor([
        [# row 0
            [# top left
                [[0,1], [2,3]], 
                [[8,9], [10,11]]
            ], 
            [# top right
                [[4,5], [6,7]], 
                [[12,13], [14,15]]
            ]  
        ],
        [# row 1
            [# bottom left
                [[16,17], [18,19]], 
                [[24,25], [26,27]]
            ],
            [# bottom right
                [[20,21], [22,23]], 
                [[28,29], [30,31]]
            ]
        ]
        
    ])
    expected_patches_image1 = expected_patches_image0 + 100
    
    # check if the first image's patches match the expected ones
    assert torch.all(patches[0] == expected_patches_image0)
    assert torch.all(patches[1] == expected_patches_image1)

def test_patchify_with_overlap():
    image = torch.Tensor(1, 4, 4, 1) # batch of 1 image, 4x4 size, 1 channel
    image[0] = torch.arange(4*4).reshape(4, 4, 1)
    #[[(0), (1), (2), (3)],
    # [(4), (5), (6), (7)],
    # [(8), (9), (10), (11)],
    # [(12), (13), (14), (15)]]
    
    grid_shape = (2, 2) # grid of 2x2 patches
    
    # OVERLAP 1
    
    patches = patchify(image, grid_shape, overlap=1)
    assert patches.shape == (1, 2, 2, 3, 3, 1) # (batch_size, n_patches_x, n_patches_y, patch_height, patch_width, channels)

    # fabricate expected patches for the image with overlap
    expected_patches = torch.tensor([
        [# row 0
            [# top left
                [[0], [1], [2]],
                [[4], [5], [6]],
                [[8], [9], [10]]
            ],
            [# top right
                [[2], [3], [0]],
                [[6], [7], [0]],
                [[10], [11], [0]]
            ]
        ],
        [# row 1
            [# bottom left
                [[8], [9], [10]],
                [[12], [13], [14]],
                [[0], [0], [0]]
            ],
            [# bottom right
                [[10], [11], [0]],
                [[14], [15], [0]],
                [[0], [0], [0]]
            ]
        ]
    ])
    assert torch.all(patches[0] == expected_patches)

    # OVERLAP 2 (creates zero-padding)

    patches = patchify(image, grid_shape, overlap=2)
    assert patches.shape == (1, 2, 2, 4, 4, 1) # (batch_size, grid_rows, grid_cols, patch_height, patch_width, channels)

    # fabricate expected patches for the image with overlap 2
    expected_patches_overlap2 = torch.tensor([
        [# row 0
            [# top left
                [[0], [0], [0], [0]],
                [[0], [0], [1], [2]],
                [[0], [4], [5], [6]],
                [[0], [8], [9], [10]]
            ],
            [# top right
                [[0], [0], [0], [0]],
                [[1], [2], [3], [0]],
                [[5], [6], [7], [0]],
                [[9], [10], [11], [0]]
            ]
        ],
        [# row 1
            [# bottom left
                [[0], [4], [5], [6]],
                [[0], [8], [9], [10]],
                [[0], [12], [13], [14]],
                [[0], [0], [0], [0]]
            ],
            [# bottom right
                [[5], [6], [7], [0]],
                [[9], [10], [11], [0]],
                [[13], [14], [15], [0]],
                [[0], [0], [0], [0]]
            ]
        ]
    ])
    assert torch.all(patches[0] == expected_patches_overlap2)


def test_unpatchify_grid():
    patch_side = 32
    base_patch = torch.arange(patch_side * patch_side).reshape(patch_side, patch_side)

    # Arrange four distinct patches in a 2x2 grid for each image.
    patches_image0 = torch.stack([
        torch.stack([base_patch, base_patch + 2000]),
        torch.stack([base_patch + 4000, base_patch + 6000]),
    ])
    patches = torch.stack([patches_image0, patches_image0 + 10000])

    images = unpatchify_grid(patches)
    assert images.shape == (2, 64, 64)

    expected_image0 = torch.cat([
        torch.cat([base_patch, base_patch + 2000], dim=1),
        torch.cat([base_patch + 4000, base_patch + 6000], dim=1),
    ], dim=0)
    assert torch.equal(images[0], expected_image0)
    assert torch.equal(images[1], expected_image0 + 10000)
