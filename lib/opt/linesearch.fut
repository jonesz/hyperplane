-- Algorithm 3.1 (Backtracking Line Search).
def backtracking obj f_k x_k p_k max_iter = 
	let a_hat = 1f32    -- 1 for N/QN.
	let rho   = 0.33f32 -- rho \in (0, 1)
	let c     = 0.95f32 -- c \in (0, 1)

	let cond a =
		-- obj (x_k + a * p_k)
		let left  = map (f32.* a) p_k |> map2 (f32.+) x_k |> obj
		-- obj (x_k) + c * a * f_k^T * p_k
		let right = map2 (f32.*) f_k p_k |> reduce (f32.+) 0f32
			|> (f32.*) (c * a) |> (f32.+) (obj x_k)

		in left <= right

	let (_, a_ast) = loop (k, a) = (0, a_hat) while
		(k < max_iter) && (cond a |> not) do -- repeat *until* -> not.
		(k + 1, rho * a)

	in a_ast
