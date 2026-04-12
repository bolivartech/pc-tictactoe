// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Cross-crate integration tests (CHECKPOINT 4).
//!
//! Verifies that `AppConfig` → `PcActorCritic` → game loop → save/load
//! works end-to-end.

#[cfg(test)]
mod tests {
    use pc_rl_core::pc_actor::SelectionMode;
    use pc_rl_core::pc_actor_critic::PcActorCritic;
    use pc_rl_core::serializer::{load_agent, save_agent};
    use pc_rl_core::CpuLinAlg;

    use crate::env::tictactoe::{Player, TicTacToe};
    use crate::utils::config::{AppConfig, HiddenLayerDef};

    /// Creates an agent from default AppConfig.
    fn agent_from_default_config() -> PcActorCritic {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config().unwrap();
        PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap()
    }

    #[test]
    fn test_agent_from_config_plays_complete_game_without_panic() {
        let mut agent = agent_from_default_config();
        let mut env = TicTacToe::new();

        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
            env.step(action).unwrap();
        }

        assert!(env.is_terminal());
    }

    #[test]
    fn test_latent_size_matches_critic_input_size() {
        let config = AppConfig::default();
        let latent_sum: usize = config
            .agent
            .actor
            .hidden_layers
            .iter()
            .map(|h| h.size)
            .sum();
        let expected_critic_input = config.agent.actor.input_size + latent_sum;
        assert_eq!(
            config.agent.critic.input_size,
            expected_critic_input,
            "critic.input_size ({}) != actor.input_size ({}) + latent_sum ({}) = {}",
            config.agent.critic.input_size,
            config.agent.actor.input_size,
            latent_sum,
            expected_critic_input,
        );
    }

    #[test]
    fn test_config_validation_catches_topology_inconsistency() {
        let mut config = AppConfig::default();
        config.agent.critic.input_size = 999;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_save_load_survives_full_session() {
        let mut agent = agent_from_default_config();
        let mut env = TicTacToe::new();

        // Play a complete game
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Training);
            env.step(action).unwrap();
        }

        // Save
        let dir = std::env::temp_dir().join(format!("pc_integ_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("session.json");
        let path_str = path.to_string_lossy().to_string();
        save_agent(&agent, &path_str, 1, None).unwrap();

        // Load
        let (loaded, metadata) = load_agent(&path_str, CpuLinAlg::new()).unwrap();
        assert_eq!(metadata.episode, 1);
        assert_eq!(
            loaded.config.actor.input_size,
            agent.config.actor.input_size
        );
        assert_eq!(
            loaded.config.critic.input_size,
            agent.config.critic.input_size
        );

        // Loaded agent can still play
        let mut env2 = TicTacToe::new();
        let mut loaded_agent = loaded;
        while !env2.is_terminal() {
            let state = env2.board_as_f64(env2.current_player());
            let valid = env2.valid_actions();
            let (action, _) = loaded_agent.act(&state, &valid, SelectionMode::Play);
            env2.step(action).unwrap();
        }
        assert!(env2.is_terminal());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_toml_two_hidden_layers_correct_critic_input_fails_if_wrong() {
        let mut config = AppConfig::default();
        config.agent.actor.hidden_layers = vec![
            HiddenLayerDef {
                size: 18,
                activation: "tanh".to_string(),
            },
            HiddenLayerDef {
                size: 12,
                activation: "relu".to_string(),
            },
        ];
        // Wrong critic input: still 27 instead of 9 + 18 + 12 = 39
        config.agent.critic.input_size = 27;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_two_hidden_layers_correct_input_passes() {
        let mut config = AppConfig::default();
        config.agent.actor.hidden_layers = vec![
            HiddenLayerDef {
                size: 18,
                activation: "tanh".to_string(),
            },
            HiddenLayerDef {
                size: 12,
                activation: "relu".to_string(),
            },
        ];
        // Correct critic input: 9 + 18 + 12 = 39
        config.agent.critic.input_size = 39;
        assert!(config.validate().is_ok());

        // Also verify we can create an agent from this config
        let agent_config = config.to_agent_config().unwrap();
        let mut agent: PcActorCritic =
            PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();

        // Play a game to verify topology works end-to-end
        let mut env = TicTacToe::new();
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
            env.step(action).unwrap();
        }
        assert!(env.is_terminal());
    }

    // ── CL (Continuous Learning) Integration Tests ──────────────────────

    /// Creates a CL-enabled agent from config with hysteresis enabled.
    fn cl_agent_from_config() -> PcActorCritic {
        let mut config = AppConfig::default();
        config.agent.actor_hysteresis = true;
        config.agent.critic_hysteresis = true;
        config.agent.scale_floor = 0.0;
        config.agent.ewc_lambda = 0.1;
        config.agent.consolidation_decay = 0.95;
        let agent_config = config.to_agent_config().unwrap();
        PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap()
    }

    #[test]
    fn test_cl_agent_plays_complete_game() {
        let mut agent = cl_agent_from_config();
        let mut env = TicTacToe::new();
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play);
            env.step(action).unwrap();
        }
        assert!(env.is_terminal());
    }

    #[test]
    fn test_cl_save_load_preserves_config() {
        let agent = cl_agent_from_config();
        let dir = std::env::temp_dir().join(format!("pc_cl_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("cl_session.json");
        let path_str = path.to_string_lossy().to_string();

        save_agent(&agent, &path_str, 100, None).unwrap();
        let (loaded, metadata) = load_agent(&path_str, CpuLinAlg::new()).unwrap();

        assert_eq!(metadata.episode, 100);
        assert!(loaded.config.actor_hysteresis);
        assert!(loaded.config.critic_hysteresis);
        assert!((loaded.config.scale_floor - 0.0).abs() < 1e-12);
        assert!((loaded.config.ewc_lambda - 0.1).abs() < 1e-12);
        assert!((loaded.config.consolidation_decay - 0.95).abs() < 1e-12);

        // Loaded CL agent can still play
        let mut loaded_agent = loaded;
        let mut env = TicTacToe::new();
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = loaded_agent.act(&state, &valid, SelectionMode::Play);
            env.step(action).unwrap();
        }
        assert!(env.is_terminal());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_champion_finder_e2e_generates_valid_file() {
        use crate::training::champion::ChampionFinder;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        let mut config = AppConfig::default();
        config.training.episodes = 30;
        config.continuous.max_episodes = 30;
        config.training.log_interval = 0;
        config.champion.n_iterations = 2;
        config.champion.assessment_games_running = 3;
        config.champion.assessment_games_final = 5;
        config.champion.assessment_interval = 15;
        config.champion.assessment_depth = 1;
        let output = format!(
            "{}/e2e_champion_{}.json",
            std::env::temp_dir().display(),
            std::process::id()
        );
        config.champion.output_path = output.clone();

        let stop = Arc::new(AtomicBool::new(false));
        let mut finder = ChampionFinder::new(config, stop);
        let result = finder.find().unwrap();

        assert_eq!(result.iterations.len(), 2);
        assert!(std::path::Path::new(&output).exists());

        // Champion should be loadable
        let (_agent, _meta) =
            pc_rl_core::serializer::load_agent(&output, pc_rl_core::CpuLinAlg::new()).unwrap();

        let _ = std::fs::remove_file(&output);
    }

    #[test]
    fn test_stress_test_e2e_generates_valid_csv() {
        use crate::training::stress_test::StressTester;
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        // First create a champion to load
        let mut config = AppConfig::default();
        config.training.log_interval = 0;
        let agent_cfg = config.to_agent_config().unwrap();
        let agent = pc_rl_core::pc_actor_critic::PcActorCritic::new(
            pc_rl_core::CpuLinAlg::new(),
            agent_cfg,
            42,
        )
        .unwrap();
        let champion_path = format!(
            "{}/e2e_stress_champion_{}.json",
            std::env::temp_dir().display(),
            std::process::id()
        );
        pc_rl_core::serializer::save_agent(&agent, &champion_path, 0, None).unwrap();

        let csv_path = format!(
            "{}/e2e_stress_log_{}.csv",
            std::env::temp_dir().display(),
            std::process::id()
        );
        let post_path = format!(
            "{}/e2e_stress_post_{}.json",
            std::env::temp_dir().display(),
            std::process::id()
        );

        let mut stress_config = config.stress_test.clone();
        stress_config.champion_path = champion_path.clone();
        stress_config.max_episodes = 20;
        stress_config.assessment_interval = 10;
        stress_config.assessment_games = 3;
        stress_config.log_path = csv_path.clone();
        stress_config.output_agent_path = post_path.clone();
        stress_config.opponent_depth_min = 1;
        stress_config.opponent_depth_max = 2;

        let stop = Arc::new(AtomicBool::new(false));
        let mut tester = StressTester::new(config, stress_config, stop).unwrap();
        let result = tester.run().unwrap();

        assert_eq!(result.total_episodes, 20);
        assert!(std::path::Path::new(&csv_path).exists());
        assert!(std::path::Path::new(&post_path).exists());

        let csv_content = std::fs::read_to_string(&csv_path).unwrap();
        assert!(csv_content.starts_with("episode,opponent_depths_seen,fitness"));
        assert!(csv_content.contains("BASELINE"));

        // Post-stress agent should be loadable
        let (_a, _m) =
            pc_rl_core::serializer::load_agent(&post_path, pc_rl_core::CpuLinAlg::new()).unwrap();

        let _ = std::fs::remove_file(&champion_path);
        let _ = std::fs::remove_file(&csv_path);
        let _ = std::fs::remove_file(&post_path);
    }

    #[test]
    fn test_step_masked_completes_episode() {
        let mut agent = cl_agent_from_config();
        let mut env = TicTacToe::new();
        agent.reset_step();

        let mut steps = 0;
        while !env.is_terminal() {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let action = agent.step_masked(&state, &valid, 0.0, false).unwrap();
            steps += 1;
            env.step(action).unwrap();
        }

        // Terminal step
        let final_state = env.board_as_f64(Player::One);
        let final_valid: Vec<usize> = (0..9).collect();
        let _ = agent.step_masked(&final_state, &final_valid, 0.0, true);

        assert!(steps > 0);
        assert!(env.is_terminal());
    }
}
