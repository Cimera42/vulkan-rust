

pub fn vec_size<T>(vector: &Vec<T>) -> usize {
    return vector.len() * std::mem::size_of::<T>();
}
