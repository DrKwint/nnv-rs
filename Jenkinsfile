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
                sh 'RUST_LOG=trace $CARGO_HOME/bin/cargo bench'
                sh '$CARGO_HOME/bin/cargo bench'
            }
        }
    }
}
