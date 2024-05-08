import "../github.com/diku-dk/sparse/compressed"
import "../github.com/diku-dk/linalg/linalg"

local module type newton = {
	type t
	type~ H [m] -- A square, inverse Hessian (B^{-1}).

	val pk [m] : H[m] -> [m]t -> [m]t

	val bfgs_update [m] : (H_k: H[m]) -> (s_k: [m]t) -> (y_k: [m]t) -> H[m]
}

module mk_newton(T: real) : newton with t = T.t = {
	type t = T.t

	module linalg_f32 = mk_linalg T
	type~ H[m] = [m][m]t

	-- (6.2) p_k = -1 * H_k * f_k
	def pk H_k f_k = 
		linalg_f32.matvecmul_row H_k f_k |> linalg_f32.vecscale (T.f32 <| -1f32)

	def bfgs_update [m] H_k (s_k: [m]t) (y_k: [m]t) =
		let I = linalg_f32.eye m

		-- (6.14) p_k = 1 / y_k^T * s_k
		let rho_k = map2 (T.*) y_k s_k |> reduce (T.+) (T.i32 0) |> (T./) (T.i32 1)

		-- (6.17) H_k1 = (I - rho_k * s_k * y_k^T) * H_k
		--   * (I - rho_k * y_k * s_k^T) 
		--   + rho_k * s_k * s_k^T
		let H_k1_left = 
			linalg_f32.outer s_k y_k |> linalg_f32.matscale rho_k |> linalg_f32.matsub I
		let H_k1_right =
			linalg_f32.outer y_k s_k |> linalg_f32.matscale rho_k |> linalg_f32.matsub I
		let H_k1_final = 
			linalg_f32.outer s_k s_k |> linalg_f32.matscale rho_k
	
		in linalg_f32.matmul (linalg_f32.matmul H_k1_left H_k) H_k1_right |> linalg_f32.matadd H_k1_final
}

-- module mk_newton_sparse(T: real) : newton with t = T.t = {
-- 	type t = T.t
-- 
-- 	module compressed_f32 = mk_compressed T
-- 	type~ mat[n][m] = compressed_f32.sc[n][m]
-- }
