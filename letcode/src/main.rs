mod tests;

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
  pub val: i32,
  pub next: Option<Box<ListNode>>
}

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

    pub fn missing_number(nums: Vec<i32>) -> i32 {
        let mut cloned = nums.clone();
        cloned.sort();
        
        for (i, val) in cloned.iter().enumerate() {
            if i as i32 != *val {
                return i as i32;
            }
        }
        
        cloned[cloned.len() - 1] + 1
    }

    pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
        let mut curr = head.as_ref();
        let mut arr: Vec<i32> = vec![];
        while let Some(curr_node) = curr {
            arr.push(curr_node.val);
            curr = curr_node.next.as_ref();
        }
        
        let mut i = 0;
        while i < arr.len()/2 {
            if arr[i] != arr[arr.len()-1 - i] {
                return false;
            }
            i += 1;
        }
        true
    }
}



fn main() {
}

