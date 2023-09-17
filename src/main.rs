use std::fs::read_to_string;
use std::path::Path;

use clap::{crate_authors, crate_description, crate_version, Arg, Command};
use gc_arena::rootless_arena;

use petal2::{compile, interpret, timed};

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
    let path = Path::new(path);
    if !path.exists() {
        panic!("Path '{}' does not exist", path.display());
    }

    let file_name = path.file_name().map(|path| path.to_string_lossy());
    let source = read_to_string(path).expect("Could not read file");
    compile_and_run(file_name.as_ref().map(|f| f.as_ref()), source);
}

fn compile_and_run<S: AsRef<[u8]>>(file_name: Option<&str>, source: S) {
    if let Some(file_name) = file_name {
        println!("File: {}", file_name);
    }

    rootless_arena(|mc| {
        let (r, ct) = timed(|| compile(mc, |m| println!("{}", m), source));
        let function = match r {
            Ok(prototype) => prototype,
            Err(error) => {
                println!("{}", error);
                return;
            }
        };

        let (r, it) = timed(|| interpret(mc, function));
        match r {
            Ok(result) => {
                println!("Compile time: {:?}", ct);
                println!("Result: {:?}", result);
                println!("Interpret time: {:?}", it);
            }
            Err(error) => {
                println!("Error occurred while interpreting: {:?}", error);
            }
        }
    })
}
