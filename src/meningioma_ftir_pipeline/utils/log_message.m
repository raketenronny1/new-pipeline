function log_message(message, log_file)
    % LOG_MESSAGE Write a timestamped message to both console and log file
    %   log_message(message, log_file) writes the message with a timestamp
    %   to both the console and the specified log file if provided
    
    % Format timestamp
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    
    % Ensure message is a string and escape special characters
    message = regexprep(char(message), '\\', '\\\\');  % Escape backslashes
    
    % Create formatted message
    formatted_msg = sprintf('[%s] %s\n', timestamp, message);
    
    % Write to console
    fprintf(formatted_msg);
    
    % Write to log file if provided
    if nargin > 1 && ~isempty(log_file)
        fprintf(log_file, formatted_msg);
    end
end