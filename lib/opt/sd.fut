import "../github.com/diku-dk/linalg/linalg"
import "linesearch"

module linalg_f32 = mk_linalg f32

let grad f x = vjp f x 1f32

-- Basic Steepest Descent.
def sd obj x_0 max_iter tol =
	let (_, x_ast, _) = loop (k, x_k, f_k) = (0, x_0, (grad obj x_0)) while
		(k < max_iter) && (linalg_f32.vecnorm f_k > tol) do

		let p_k = map (f32.* -1f32) f_k
		let a_k = backtracking obj x_k p_k 1000

		let x_k1 = map (f32.* a_k) p_k |> map2 (f32.+) x_k

		in (k + 1, x_k1, grad obj x_k1)

	in x_ast
	
