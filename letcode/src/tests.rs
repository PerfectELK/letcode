use crate::{ListNode, Solution};

#[test]
fn convert_to_title(){
    struct TestCase {
        column_number: i32,
        result: String,
    }

    let cases = vec![
        TestCase{column_number: 0, result: "".to_string()},
        TestCase{column_number: 1, result: "A".to_string()},
        TestCase{column_number: 26, result: "Z".to_string()},
        TestCase{column_number: 52, result: "AZ".to_string()},
        TestCase{column_number: 28, result: "AB".to_string()},
        TestCase{column_number: 701, result: "ZY".to_string()},
        TestCase{column_number: 703, result: "AAA".to_string()},
    ];

    for test_case in cases {
        let result = Solution::convert_to_title(test_case.column_number);
        assert_eq!(
            test_case.result, result,
            "Input {}", test_case.column_number
        );
    }
}

#[test]
fn title_to_number(){
    struct TestCase {
        column_title: String,
        result: i32,
    }  
    
    let cases = vec![
        TestCase{column_title: "A".to_string(), result: 1,},
        TestCase{column_title: "Z".to_string(), result: 26,},
        TestCase{column_title: "AZ".to_string(), result: 52,},
        TestCase{column_title: "AAA".to_string(), result: 703,},
    ];
    
    for test_case in cases {
        let result = Solution::title_to_number(test_case.column_title.clone());
        assert_eq!(
            test_case.result, result,
            "Input {}", test_case.column_title
        );
    }
}

#[test]
fn is_power_of_two(){
    struct TestCase {
        n: i32,
        result: bool,
    }
    
    let cases = vec![
        TestCase{n: 1, result: true,},
        TestCase{n: 2, result: true,},
        TestCase{n: 3, result: false,},
        TestCase{n: 16, result: true,},
        TestCase{n: 38, result: false,},
    ];
    
    for test_case in cases {
        let result = Solution::is_power_of_two(test_case.n);
        assert_eq!(
            test_case.result, result,
            "Input {}", test_case.n
        )
    }

}

#[test]
fn missing_number(){
    struct TestCase {
        nums: Vec<i32>,
        result: i32,
    }
    
    let cases = vec![
        TestCase{
            nums: vec![3,0,1],
            result: 2,
        },        
        TestCase{
            nums: vec![0,1],
            result: 2,
        },        
        TestCase{
            nums: vec![9,6,4,2,3,5,7,0,1],
            result: 8,
        },
    ];
    
    for test_case in cases {
        let result = Solution::missing_number(test_case.nums.clone());
        assert_eq!(
            test_case.result, result,
            "Input {:?}", test_case.nums
        )
    }

}
#[test]
fn is_palindrome() {
    struct TestCase {
        head:  Option<Box<ListNode>>,
        result: bool,
    }

    let cases = vec![
        TestCase{
            head: Some(Box::new(
                ListNode{
                    val:1,
                    next: Some(Box::new(ListNode{
                        val:2,
                        next: Some(Box::new(ListNode{
                            val: 2,
                            next: Some(Box::new(
                            ListNode{
                                val: 1,
                                next: None,
                            }
                        ))
                    })),
                }))})),
            result: true,
        },
        TestCase{
            head: Some(Box::new(
                ListNode{
                    val:1,
                    next: Some(Box::new(ListNode{
                        val:2,
                        next: None,
                }))})),
            result: false,
        },
    ];

    for test_case in cases {
        let result = Solution::is_palindrome(test_case.head.clone());
        assert_eq!(
            test_case.result, result,
            "Input {:?}", test_case.head
        )
    }
}