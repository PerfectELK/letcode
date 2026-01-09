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

    pub fn title_to_number(column_title: String) -> i32 {
        let reversed: String = column_title.chars().rev().collect();
        
        let mut accumulator: i32 = 0;
        for (i, c) in reversed.chars().enumerate() {
            let num = c as i32 - 64;
            accumulator = accumulator + (26_i32.pow(i as u32) * num);
        }
        
        accumulator
    }

    pub fn is_power_of_two(n: i32) -> bool {
        if n < 1 {
            return false;
        }
        n & (n -1) == 0
    }
}



fn main() {
}

