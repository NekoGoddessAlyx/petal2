use std::fs::read_to_string;
use std::path::Path;

use clap::{crate_authors, crate_description, crate_version, Arg, Command};
use gc_arena::rootless_arena;

use petal2::{compile, interpret, CompilerMessage, PString};

fn callback(message: CompilerMessage<PString>) {
    println!("{}", message);
}

fn main() {
    let command = Command::new("petal")
        .arg_required_else_help(true)
        .about(crate_description!())
        .author(crate_authors!())
        .version(crate_version!())
        .disable_version_flag(true)
        .arg(
            // all for lowercase v
            Arg::new("version")
                .help("Print version")
                .short('v')
                .long("version")
                .action(clap::ArgAction::Version),
        )
        .arg(
            Arg::new("file")
                .help("File to interpret")
                .num_args(1)
                .index(1)
                .required(true),
        );

    let matches = command.get_matches();

    let path = matches.get_one::<String>("file").expect("File is required");
    let path = Path::new(&path);
    if !path.exists() {
        panic!("Path '{}' does not exist", path.display());
    }

    let source = read_to_string(path).expect("Could not read file");

    rootless_arena(|mc| {
        let compile_result = compile(mc, callback, source);
        let function = match compile_result {
            Ok(prototype) => {
                println!("Prototype: {}", prototype);
                prototype
            }
            Err(error) => {
                println!("{}", error);
                return;
            }
        };

        match interpret(mc, function) {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(error) => {
                println!("Error occurred while interpreting: {:?}", error);
            }
        }
    })
}
