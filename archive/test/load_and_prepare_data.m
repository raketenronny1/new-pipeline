function load_and_prepare_data(cfg)
    % Patch function to call the fixed implementation
    % This overrides the original load_and_prepare_data in the test environment
    load_and_prepare_data_fixed(cfg);
end
