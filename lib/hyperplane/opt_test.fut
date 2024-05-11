-- | ignore

import "linesearch"
import "sd"
import "newton"

let rosenbrock [m] (x: [m]f32) : f32 = 
	(1f32 - x[0]) ** 2 + 100f32 * (x[1] - x[0]**2)**2

module qn = mk_bfgs f32 (mk_backtracking f32)
module sd = mk_sd f32   (mk_backtracking f32)

-- ==
-- entry: test_sd
-- input { [0.0f32, 5.0f32] }
-- output { [1.0f32, 1.0f32] }

entry test_sd x_0 = 
	sd.iter rosenbrock x_0 1000i64 1e-3f32

-- ==
-- entry: test_bfgs
-- input { [0.0f32, 5.0f32] }
-- output { [1.0f32, 1.0f32] }

entry test_bfgs x_0 = 
	qn.iter rosenbrock x_0 10000i64 1e-3f32
