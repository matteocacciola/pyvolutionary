import numpy as np

from .models import Gene


class BitGene:
    def __init__(self, space_dimension: int):
        self.__space_dimension = space_dimension
        self.__n_bits = 8 * space_dimension
        self.__bitstring = np.random.randint(0, 2, self.__n_bits * space_dimension).tolist()

    @property
    def n_bits(self) -> int:
        return self.__n_bits

    @property
    def bitstring(self) -> list[int]:
        return self.__bitstring

    def decode(self, lower_bound: list[float], upper_bound: list[float]) -> list[float]:
        """
        Decode a bitstring to a float list.
        :param lower_bound: the lower bound of the search space
        :param upper_bound: the upper bound of the search space
        :return: the decoded float list
        :rtype: list[float]
        """
        decoded = []
        largest = 2 ** self.n_bits

        lb, ub = np.array(lower_bound), np.array(upper_bound)
        for i in range(0, self.__space_dimension):
            # extract the substring
            start, end = i * self.n_bits, (i * self.n_bits) + self.n_bits
            substring = self.bitstring[start:end]
            # convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            # convert string to integer
            integer = int(chars, 2)
            # scale integer to desired range
            value = lb[i] + (integer / largest) * (ub[i] - lb[i])
            # store
            decoded.append(value)
        return decoded

    def encode(self, position: list[float], lower_bound: list[float], upper_bound: list[float]) -> "BitGene":
        """
        Encode a float list to a bitstring.
        :param position: the float list to encode
        :param lower_bound: the lower bound of the search space
        :param upper_bound: the upper bound of the search space
        :return: the encoded bitstring
        :rtype: list[int]
        """
        encoded = []

        lb, ub = np.array(lower_bound), np.array(upper_bound)
        for i in range(0, self.__space_dimension):
            # scale integer to desired range
            value = (position[i] - lb[i]) / (ub[i] - lb[i])
            # convert to bitstring
            value = bin(int(value * (2 ** self.n_bits)))[2:].zfill(self.n_bits)
            binary_representation = [int(bit) for bit in value]
            # append to the encoded list
            encoded.extend(binary_representation)
        self.__bitstring = encoded
        return self

    def set_bit_string(self, bitstring: list[int]) -> "BitGene":
        self.__bitstring = bitstring
        return self


class Transformer:
    @staticmethod
    def from_bit_gene_to_gene(bit_gene: BitGene, lower_bounds: list[float], upper_bounds: list[float], gene_init_fnc: callable) -> Gene:
        return Gene(**gene_init_fnc(bit_gene.decode(lower_bounds, upper_bounds)).model_dump())
