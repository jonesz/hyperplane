module type method = {
	type t
	val step [m] : (obj: [m]t -> t) -> (x_0: [m]t) -> (max_iter: i64) -> (tol: t) -> [m]t
}
