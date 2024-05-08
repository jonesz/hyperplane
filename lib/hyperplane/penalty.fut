module type penalty = {
	type t 

	val ecq [m] : ([m]t -> t) -> [m]t -> (u: t) -> t
	val ecl [m] : ([m]t -> t) -> [m]t -> (u: t) -> (l: t) -> t

	-- val icq [m] : ([m]t -> t) -> [m]t -> (u: t) -> t
	-- val icl [m] : ([m]t -> t) -> [m]t -> (u: t) -> (l: t) -> t
}

module mk_penalty(T: real) : penalty with t = T.t = {
	type t = T.t

	def ecq f x u =
		-- u/2 * (c_i(x))^2
		(T.*) (f x) (f x) |> (T.*) (T.f32 0.5f32) |> (T.*) u

	def ecl f x u l =
		let q = (T.*) (f x) (f x) |> (T.*) (T.f32 0.5f32) |> (T.*) u
		let l = (T.*) (f x) (T.f32 (-1.0f32)) |> (T.*) l
		in (T.+) q l
}
