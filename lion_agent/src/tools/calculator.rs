// lion_agent/src/tools/calculator.rs — Math expression evaluator

use std::pin::Pin;
use std::future::Future;
use crate::tool::{Tool, ToolResult};

use evalexpr::{ContextWithMutableFunctions, HashMapContext, Value, Function, eval_with_context, EvalexprError};

pub struct Calculator;

/// Helper to convert evalexpr::Value to f64 regardless of whether it was parsed as Float or Int
fn val_to_f64(val: &Value) -> Result<f64, EvalexprError> {
    match val {
        Value::Float(f) => Ok(*f),
        Value::Int(i)   => Ok(*i as f64),
        _ => Err(EvalexprError::ExpectedNumber { actual: val.clone() }),
    }
}

impl Tool for Calculator {
    fn name(&self)         -> &'static str { "calculator" }
    fn description(&self)  -> &'static str { "Evaluates a mathematical expression including standard functions (sqrt, sin, cos, abs)" }
    fn input_format(&self) -> &'static str { "A math expression e.g.: sqrt(1764) * 5 or 2 + 2 * 10" }

    fn execute<'a>(&'a self, input: &'a str) -> Pin<Box<dyn Future<Output = ToolResult> + Send + 'a>> {
        Box::pin(async move {
            let expr = input.trim()
                .replace('×', "*")
                .replace('÷', "/")
                .replace('^', " ^ ");

            let mut context = HashMapContext::new();

            // Bind sqrt(x)
            let _ = context.set_function("sqrt".to_string(), Function::new(|arg| {
                let num = val_to_f64(arg)?;
                Ok(Value::from(num.sqrt()))
            }));

            // Bind abs(x)
            let _ = context.set_function("abs".to_string(), Function::new(|arg| {
                let num = val_to_f64(arg)?;
                Ok(Value::from(num.abs()))
            }));

            // Bind sin(x)
            let _ = context.set_function("sin".to_string(), Function::new(|arg| {
                let num = val_to_f64(arg)?;
                Ok(Value::from(num.sin()))
            }));

            // Bind cos(x)
            let _ = context.set_function("cos".to_string(), Function::new(|arg| {
                let num = val_to_f64(arg)?;
                Ok(Value::from(num.cos()))
            }));

            // Bind tan(x)
            let _ = context.set_function("tan".to_string(), Function::new(|arg| {
                let num = val_to_f64(arg)?;
                Ok(Value::from(num.tan()))
            }));

            match eval_with_context(&expr, &context) {
                Ok(v)  => ToolResult::ok(format!("{} = {}", input.trim(), v)),
                Err(e) => {
                    // Fallback to simple eval if context functions weren't matched or if it was simple
                    match evalexpr::eval(&expr) {
                        Ok(v_simple) => ToolResult::ok(format!("{} = {}", input.trim(), v_simple)),
                        Err(_) => ToolResult::err(format!("Cannot evaluate '{}': {}", input.trim(), e)),
                    }
                }
            }
        })
    }
}
