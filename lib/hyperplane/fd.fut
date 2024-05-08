def grad f x = vjp f x 1f32

def hess_approx_fwd_fd [m] obj (x: [m]f32) (h: [m]f32) e =
	let f_k = grad obj x

	in map2 (\i h_i -> 
		let x_eps = copy x with [i] = x[i] + h_i * e
		let f_k_eps = grad obj x_eps
		in map2 (f32.-) f_k_eps f_k |> map (f32./ h_i)
	) (iota m) h
