module type penalty = {
	type t 

	val ecq [m] : ([m]t -> t) -> [m]t -> (u: t) -> t
	val ecl [m] : ([m]t -> t) -> [m]t -> (u: t) -> (l: t) -> t

	val icq [m] : ([m]t -> t) -> [m]t -> (u: t) -> t
	val icl [m] : ([m]t -> t) -> [m]t -> (u: t) -> (l: t) -> t

	-- Compute the next lagrange multiplier for the constraint.
	val lm_next [m] : ([m]t -> t) -> [m]t -> (u: t) -> (l: t) -> t
}

module mk_penalty(T: real) : penalty with t = T.t = {
	type t = T.t

	let q c x =
		(T.*) (c x) (c x)

	let mul_u2 u =
		(T.*) u (T.f32 0.5f32) |> (T.*)

	def ecq c x u =
		-- u/2 * (c_i(x))^2
		q c x |> mul_u2 u

	def ecl c x u l =
		let q = q c x |> mul_u2 u
		let l = (T.*) (c x) (T.f32 (-1.0f32)) |> (T.*) l
		in (T.+) q l

	def icq c x u =
		-- u/2 * (max(-1 * c_i(x), 0))^2
		let y = c x |> (T.*) (T.f32 (-1.0f32)) |> T.max (T.i32 0i32)
		in (T.*) y y |> mul_u2 u

	def icl _c _x _u _l =
		???

	def lm_next c x u l =
		-- (17.39) l_i^{k+1} = l_i^k - u_k * c_i(x_k).
		c x |> (T.*) u |> (T.-) l
}
