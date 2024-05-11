import "../github.com/diku-dk/sparse/compressed"
import "../github.com/diku-dk/linalg/linalg"
import "method"
import "linesearch"

module mk_bfgs (T: real) (L: linesearch with t = T.t): method with t = T.t = {
	type t = T.t
	module linalg_t = mk_linalg T

	let grad f x = vjp f x (T.f32 1f32)

	-- (6.2) p_k = -1 * H_k * f_k
	let pk H_k f_k = 
		linalg_t.matvecmul_row H_k f_k |> linalg_t.vecscale (T.f32 <| -1f32)

	let bfgs_update [m] H_k (s_k: [m]t) (y_k: [m]t) =
		let I = linalg_t.eye m

		-- (6.14) p_k = 1 / y_k^T * s_k
		let rho_k = map2 (T.*) y_k s_k |> reduce (T.+) (T.i32 0) |> (T./) (T.i32 1)

		-- (6.17) H_k1 = (I - rho_k * s_k * y_k^T) * H_k
		--   * (I - rho_k * y_k * s_k^T) 
		--   + rho_k * s_k * s_k^T
		let H_k1_left = 
			linalg_t.outer s_k y_k |> linalg_t.matscale rho_k |> linalg_t.matsub I
		let H_k1_right =
			linalg_t.outer y_k s_k |> linalg_t.matscale rho_k |> linalg_t.matsub I
		let H_k1_final = 
			linalg_t.outer s_k s_k |> linalg_t.matscale rho_k
	
		in linalg_t.matmul (linalg_t.matmul H_k1_left H_k) H_k1_right |> linalg_t.matadd H_k1_final

	def iter [m] (obj: [m]t -> t) x_0 max_iter tol =
		let I = linalg_t.eye m

		let (_, x_ast, _, _) = loop (k, x_k, f_k, H_k) = (0i64, x_0, (grad obj x_0), I) while
			(k < max_iter) && ((T.>) (linalg_t.vecnorm f_k) tol) do

			-- (6.2) p_k = -1 * H_k * f_k
			let p_k = pk H_k f_k
			let a_k = L.alpha obj x_k p_k 20

			let x_k1 = map (T.* a_k) p_k |> map2 (T.+) x_k

			-- (6.5) s_k = a_k * p_k
			let s_k = map2 (T.-) x_k1 x_k
			-- (6.5) y_k = f_k1 - f_k
			let y_k = map2 (T.-) (grad obj x_k1) (grad obj x_k)

			let H_k1 = bfgs_update H_k s_k y_k 

			-- If we've hit a NaN value, reset H_k to I.
			in if any (T.isnan) x_k1
				then (k + 1, x_k,  (grad obj x_k1), I)
				else (k + 1, x_k1, (grad obj x_k1), H_k1)

		in x_ast
}

