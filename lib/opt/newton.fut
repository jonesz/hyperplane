import "../github.com/diku-dk/linalg/linalg"
import "linesearch"
import "fd"

module linalg_f32 = mk_linalg f32

let grad f x = vjp f x 1f32

def newton [m] obj (x_0: [m]f32) max_iter tol =
	let (_, x_ast, _) = loop (k, x_k, f_k) = (0, x_0, (grad obj x_0)) while
		(k < max_iter) && (linalg_f32.vecnorm f_k > tol) do

		-- (6.2) p_k = -1 * H_k * f_k
		let B_k = hess_approx_fwd_fd obj x_k (replicate m 1e-7f32) 1f32 |> linalg_f32.inv
		let p_k = linalg_f32.matvecmul_row B_k f_k |> linalg_f32.vecscale (-1f32)
		let a_k = backtracking obj f_k x_k p_k 1000

		let x_k1 = map (f32.* a_k) p_k |> map2 (f32.+) x_k

		in (k + 1, x_k1, grad obj x_k)

	in x_ast

-- BFGS.
def bfgs [m] obj (x_0: [m]f32) max_iter tol =
	let I = linalg_f32.eye m

	let (_, x_ast, _, _) = loop (k, x_k, f_k, H_k) = (0, x_0, (grad obj x_0), I) while
		(k < max_iter) && (linalg_f32.vecnorm f_k > tol) do

		-- (6.2) p_k = -1 * H_k * f_k
		let p_k = linalg_f32.matvecmul_row H_k f_k |> linalg_f32.vecscale (-1f32)
		let a_k = backtracking obj f_k x_k p_k 1000

		let x_k1 = map (f32.* a_k) p_k |> map2 (f32.+) x_k

		-- Update H_k.
		-- (6.5) s_k = a_k * p_k
		let s_k = map2 (f32.-) x_k1 x_k
		-- (6.5) y_k = f_k1 - f_k
		let y_k = map2 (f32.-) (grad obj x_k1) (grad obj x_k)

		-- (6.14) p_k = 1 / y_k^T * s_k
		let rho_k = map2 (f32.*) y_k s_k |> reduce (f32.+) 0f32 |> (f32./) 1

		-- (6.17) H_k1 = (I - rho_k * s_k * y_k^T) * H_k
		--   * (I - rho_k * y_k * s_k^T) 
		--   + rho_k * s_k * s_k^T
		let H_k1_left = 
			linalg_f32.outer s_k y_k |> linalg_f32.matscale rho_k |> linalg_f32.matsub I
		let H_k1_right =
			linalg_f32.outer y_k s_k |> linalg_f32.matscale rho_k |> linalg_f32.matsub I
		let H_k1_final = 
			linalg_f32.outer s_k s_k |> linalg_f32.matscale rho_k
	
		let H_k1 = linalg_f32.matmul (linalg_f32.matmul H_k1_left H_k) H_k1_right |> linalg_f32.matadd H_k1_final

		-- If we've hit a NaN value, reset H_k to the I.
		in if any (f32.isnan) x_k1 
			then (k + 1, x_k, grad obj x_k, I)
			else (k + 1, x_k1, grad obj x_k1, H_k1)
	
	in x_ast
