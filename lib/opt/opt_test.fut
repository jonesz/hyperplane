-- | ignore

import "sd"
import "newton"

let rosenbrock x = (1f32 - x[0]) ** 2 + 100f32 * (x[1] - x[0]**2)**2

-- ==
-- entry: test_sd
-- input { [0.0f32, 5.0f32] }
-- output { [1.0f32, 1.0f32] }

entry test_sd x_0 = 
	sd rosenbrock x_0 1000 1e-3

-- ==
-- entry: test_bfgs
-- input { [0.0f32, 5.0f32] }
-- output { [1.0f32, 1.0f32] }

entry test_bfgs x_0 = 
	bfgs rosenbrock x_0 1000 1e-3

-- ==
-- entry: test_newton
-- input { [0.0f32, 5.0f32] }
-- output { [1.0f32, 1.0f32] }

entry test_newton x_0 = 
	newton rosenbrock x_0 1000 1e-3
