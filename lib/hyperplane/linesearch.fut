import "fd"

module type linesearch = {
	type t
	val alpha [m] : ([m]t -> t) -> [m]t -> [m]t -> i64 -> t
}

module mk_backtracking (T: real) : linesearch with t = T.t = {
	type t = T.t

	let grad f x = vjp f x (T.f32 1f32)

	-- Algorithm 3.1 (Backtracking Line Search).
	let backtracking [m] (obj: [m]t -> t) x_k p_k max_iter = 
		let a_hat = (T.f32 1f32)    -- 1 for N/QN.
		let rho   = (T.f32 0.33f32) -- rho \in (0, 1)
		let c     = (T.f32 0.95f32) -- c \in (0, 1)

		-- TODO: At some point, it might be worthwhile to
		-- abstract the grad into a module (which would then
		-- be passed in the creation of this module).
		let f_k   = grad obj x_k

		let cond a =
			-- obj (x_k + a * p_k)
			let left  = map (T.* a) p_k |> map2 (T.+) x_k |> obj
			-- obj (x_k) + c * a * f_k^T * p_k
			let right = map2 (T.*) f_k p_k |> reduce (T.+) (T.f32 0f32)
				|> (T.*) c |> (T.*) a |> (T.+) (obj x_k)

			in (T.<=) left right

		let (_, a_ast) = loop (k, a) = (0i64, a_hat) while
			(k < max_iter) && (cond a |> not) do -- repeat *until* -> not.
			(k + 1, (T.*) rho a)
		in a_ast

	def alpha obj x_k p_k max_iter = 
		backtracking obj x_k p_k max_iter
}

-- Algorithm 3.5 (Line Search Algorithm)
def wolfe obj x_k p_k max_iter =
	let a_0   = 0f32
	let a_max = 10f32
	let a_1   = 1.01f32

	-- Recommendations from Nocedal and Wright for QN.
	let c_1 = -1e-3f32
	let c_2 = 0.9f32

	-- (3.54) phi(a) = f(x_k + a * p_k)
	let phi  a = map (f32.* a) p_k |> map2 (f32.+) x_k |> obj
	let phi' a = jvp phi a 1f32

	-- phi(a_cur) > phi(0) + c_1 * a_cur * phi'(0)
	let sc_1 a_cur = 
		let left  = phi a_cur
		let right = phi' 0f32 |> (f32.*) a_cur |> (f32.*) c_1 |> (f32.+) (phi 0)
		in left > right

	-- phi(a_cur) >= phi(a_prev) && i > 1
	let sc_2 a_cur a_prev i = 
		let cond = phi a_prev |> (f32.>=) (phi a_cur)
		in cond && i > 1

	-- abs(phi'(a_cur)) <= -c_2 * phi'(0)
	let sc_3 a_cur = 
		let left  = phi' a_cur |> f32.abs
		let right = phi' 0f32 |> (f32.*) c_2 |> (f32.*) (-1f32)
		in left <= right

	-- phi'(a_cur) >= 0
	let sc_4 a_cur = phi' a_cur |> (f32.< 0f32)

	let zoom _a_lo _a_hi =
		???
	
	let (_, a_ast, _) = loop (k, a_cur, a_prev) = (0, a_1, a_0) while (k < max_iter) do
		if (sc_1 a_cur || sc_2 a_cur a_prev k)
		then (max_iter, zoom a_prev a_cur, a_cur)
		else if sc_3 a_cur
		then (max_iter, a_cur, a_cur)
		else if sc_4 a_cur
		then (max_iter, zoom a_cur a_prev, a_cur)
		else (k + 1, (f32./) (a_cur f32.+ a_max) 2f32, a_cur)
	in a_ast
