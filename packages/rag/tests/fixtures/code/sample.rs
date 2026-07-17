use std::io;

/// Ownership helper
fn take_ownership(s: String) {
    println!("{s}");
}

pub struct Owner {
    name: String,
}

fn main() {
    take_ownership(String::from("hello"));
}
