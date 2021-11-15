pipeline {
    agent any

    stages {
        stage('Build') {
            environment {
                CARGO_HOME = '/usr/local/cargo/'
            }

            steps {
                sh 'bash -c "cargo build"'
            }
        }
    }
}
