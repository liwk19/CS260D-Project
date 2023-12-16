import socket
import pickle
from autodse.result import Result  # I previously shared it with you. you have it here: https://github.com/yunshengb/software-gnn/tree/master/dse_database/autodse
HOST = "10.1.3.56"  # The server's hostname or IP address (u3-fcs)
PORT = 65432        # The port used by the server
point = {"__PARA__L0": 1, "__PARA__L1": 1, "__PARA__L2":1, "__PARA__L3": 1, "__PARA__L4":65536, "__PARA__L5":1, "__PIPE__L0": "off", "__PIPE__L3": "flatten"}
kernel_name = {"arithm"}
message = {"kernel_name": kernel_name, "point": point}
msg_pickle = pickle.dumps(message)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(msg_pickle)
    data_pickle = s.recv(4096)
    if data_pickle:
        data = pickle.loads(data_pickle)
        print("Result recieved:")
        obj = data.get(list(data.keys())[0])
        if isinstance(obj, Result): print(obj.ret_code, obj.perf, obj.res_util)




def finte_diff_as_quality(self, new_result: Result, ref_result: Result) -> float:
    """Compute the quality of the point by finite difference method.
    Args:
        new_result: The new result to be qualified.
        ref_result: The reference result.
    Returns:
        The quality value (negative finite differnece). Larger the better.
    """
    def quantify_util(result: Result) -> float:
        """Quantify the resource utilization to a float number.
        util' = 5 * ceil(util / 5) for each util,
        area = sum(2^1(1/(1-util))) for each util'
        Args:
            result: The evaluation result.
        Returns:
            The quantified area value with the range (2*N) to infinite,
            where N is # of resources.
        """
        # Reduce the sensitivity to (100 / 5) = 20 intervals
        utils = [
            5 * ceil(u * 100 / 5) / 100 for k, u in result.res_util.items()
            if k.startswith('util')
        ]
        # Compute the area
        return sum([2.0**(1.0 / (1.0 - u)) for u in utils])
    ref_util = quantify_util(ref_result)
    new_util = quantify_util(new_result)
    if (new_result.perf / ref_result.perf) > 1.05:
        # Performance is too worse to be considered
        return -float('inf')
    if new_util == ref_util:
        if new_result.perf < ref_result.perf:
            # Free lunch
            return float('inf')
        # Same util but slightly worse performance, neutral
        return 0
    return -(new_result.perf - ref_result.perf) / (new_util - ref_util)