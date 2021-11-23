pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'cargo build'
            }
        }

        stage('Test') {
            steps {
                sh 'cargo test'
            }
        }

        stage('Bench') {
            steps {
                sh 'RUST_LOG=trace cargo bench'
                sh 'cargo bench'
            }
        }
    }
}
