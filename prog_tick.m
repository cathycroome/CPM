function prog_tick(total, t0)
    persistent n; if isempty(n), n = 0; end
    n = n + 1;
    % print every ~5% (and at start/end)
    step = max(1, floor(total/20));
    if mod(n, step)==0 || n==1 || n==total
        fprintf(' perm %d/%d (%.1f%%) | elapsed %.1f min\n', ...
            n, total, 100*n/total, toc(t0)/60);
    end
end
