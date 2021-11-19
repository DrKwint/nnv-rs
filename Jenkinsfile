pipeline {
    agent any

    stages {
        stage('Build') {
            environment {
                CARGO_HOME = '/usr/local/cargo'
            }

            steps {
                sh '$CARGO_HOME/bin/cargo build'
            }
        }

        stage('Test') {
            environment {
                CARGO_HOME = '/usr/local/cargo'
            }

            steps {
                sh '$CARGO_HOME/bin/cargo test'
            }
        }

        stage('Bench') {
            environment {
                CARGO_HOME = '/usr/local/cargo'
            }

            steps {
                RUST_LOG=trace sh '$CARGO_HOME/bin/cargo bench'
                sh '$CARGO_HOME/bin/cargo bench'
            }
        }
    }
}
