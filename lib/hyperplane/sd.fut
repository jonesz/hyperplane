import "../github.com/diku-dk/linalg/linalg"
import "method"
import "linesearch"

-- Steepest Descent.
module mk_sd (T: real) (L: linesearch with t = T.t): method with t = T.t = {
	type t = T.t
	module linalg_t = mk_linalg T

	let grad f x = vjp f x (T.f32 1f32)

	def iter obj x_0 max_iter tol =
		let (_, x_ast, _) = loop (k, x_k, f_k) = (0i64, x_0, (grad obj x_0)) while
			(k < max_iter) && ((T.>) (linalg_t.vecnorm f_k) tol) do
			
			let p_k = map (T.* (T.i64 (-1i64))) f_k
			let a_k = L.alpha obj x_k p_k 1000

			let x_k1 = map (T.* a_k) p_k |> map2 (T.+) x_k
			in (k + 1, x_k1, grad obj x_k1)
		in x_ast
}
