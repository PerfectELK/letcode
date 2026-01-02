mod tests;

struct Solution;
impl Solution {
    pub fn convert_to_title(column_number: i32) -> String {
        if column_number <= 0 {
            return "".to_string();
        }

        let mut num = column_number;
        let mut builder = String::new();
        while num > 0 {
            num -= 1;
            let remainder = (num % 26) as u8;
            let ch = (b'A' + remainder) as char;
            builder.insert(0, ch);
            num /= 26;
        }
        builder
    }
}


fn main() {
}

